import torch, yaml, os, sys
import torch.nn as nn
import src
from torch.utils.data import DataLoader, TensorDataset
from src import cool_plot, PlotData, save_as_gif, physical_model, min_max_normalization
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--job_id', type=str, default='')
parser.add_argument('--config_file', type=str, default='./config.yaml')
parser.add_argument('--input_2d', type=eval, default=True)


class MLP(nn.Module):
    """
        Definition of a Multi-Layer Perceptron Neuran Network (MLP).
        The neural network uses the Pytorch library
    """
    
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super(MLP, self).__init__()

        # Non-linear activation function
        activation = nn.ReLU

        # Input layer
        self.fc1 = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])

        # Hidden layers
        self.fc2 = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])

        # Output layer
        self.fc3 = nn.Linear(N_HIDDEN, N_OUTPUT)       

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x
    

def main(working_dir, args):

    # Load project parameters
    with open(args.config_file, 'r') as f: 
        config = yaml.safe_load(f)

    # Load Neural Network parameters
    NUM_EPOCHS      = config["MLP"]["num_epochs"]
    N_OUTPUT        = config["MLP"]["output_dim"]
    N_HIDDEN        = config["MLP"]["hidden_neurons"]
    N_LAYERS        = config["MLP"]["hidden_layers"]
    LEARNING_RATE   = config["MLP"]["learning_rate"]
    GAMMA           = config["MLP"]["gamma"]
    MOMENTUM        = config["MLP"]["momentum"]
    OPTIMIZER_TYPE  = config["MLP"]["optimizer"].upper()
    BATCH_SIZE      = config["MLP"]["batch_size"]
    FILE_TRAIN      = config["MLP"]["file_train"]
    FILE_VALIDATION = config["MLP"]["file_validation"]
    FILE_TEST       = config["MLP"]["file_test"]
    MODEL_PATH      = config["data"]["model_path_20"]
    USE_SCHEDULER   = config["MLP"]["use_scheduler"]

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Log some information
    logger.info(f'Job ID: {args.job_id}')
    logger.info(f'Loading configuration file {args.config_file}')
    logger.info(f'The device is set to {device}\n')
    logger.info(f'Train dataset made of files: {FILE_TRAIN}')
    logger.info(f'Validation dataset made of files: {FILE_VALIDATION}')
    logger.info(f'Test dataset made of files: {FILE_TEST}')
    logger.info("")
    logger.info(f"Input type: {args.input_2d + 1}D")
    logger.info("Training settings:")
    logger.info(f"NUM_EPOCHS={NUM_EPOCHS}")
    logger.info(f"N_OUTPUT={N_OUTPUT}")
    logger.info(f"N_HIDDEN={N_HIDDEN}")
    logger.info(f"N_LAYERS={N_LAYERS}")
    logger.info(f"LEARNING_RATE={LEARNING_RATE}")
    logger.info(f"GAMMA={GAMMA}")
    logger.info(f"MOMENTUM={MOMENTUM}")
    logger.info(f"OPTIMIZER_TYPE={OPTIMIZER_TYPE}")
    logger.info(f"BATCH_SIZE={BATCH_SIZE}")
    logger.info(f"USE_SCHEDULER={USE_SCHEDULER}")
    logger.info("")
    
    # Store parameters to test case
    with open(working_dir + '/settings.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    # Load training data
    track_dist, ABA_data, track_z_displ, Ts, vel = src.signal.load_data(
        file_index=FILE_TRAIN, config_file=args.config_file, output='acceleration',
        resampled_freq=1000, resample_method='downsample'
    )

    # Load validation data
    track_valid, ABA_data_valid, track_z_valid, _, vel_valid = src.signal.load_data(
        file_index=FILE_VALIDATION, config_file=args.config_file, output='acceleration',
        resampled_freq=1000, resample_method='downsample'
    )

    # Load test data
    track_test, ABA_data_test, track_z_test, _, vel_test = src.signal.load_data(
        file_index=FILE_TEST, config_file=args.config_file, output='acceleration',
        resampled_freq=1000, resample_method='downsample'
    )

    # Filter acceleration data: NOTE: to remove in the future?
    ABA_data = src.signal.butter_filter(
        ABA_data, cutoff_freq=[1, 20], 
        fs=1/Ts, order=4, filter_type='bandpass', 
        analog=False, filter_twice=True)

    ABA_data_valid = src.signal.butter_filter(
        ABA_data_valid, cutoff_freq=[1, 20], 
        fs=1/Ts, order=4, filter_type='bandpass', 
        analog=False, filter_twice=True)

    ABA_data_test = src.signal.butter_filter(
        ABA_data_test, cutoff_freq=[1, 20], 
        fs=1/Ts, order=4, filter_type='bandpass', 
        analog=False, filter_twice=True)

    # Convert input and output vector to tensors
    u_train = torch.tensor(np.vstack([ABA_data, vel, ]).T, dtype=torch.float32, requires_grad=True).unsqueeze(1).to(device)
    u_valid = torch.tensor(np.vstack([ABA_data_valid, vel_valid, ]).T, dtype=torch.float32).unsqueeze(1).to(device)
    u_test  = torch.tensor(np.vstack([ABA_data_test, vel_test, ]).T, dtype=torch.float32).unsqueeze(1).to(device)

    y_train = torch.tensor(track_z_displ, dtype=torch.float32, requires_grad=True).reshape([-1, 1, 1]).to(device)
    y_valid = torch.tensor(track_z_valid, dtype=torch.float32).reshape([-1, 1, 1]).to(device)
    y_test  = torch.tensor(track_z_test, dtype=torch.float32).reshape([-1, 1, 1]).to(device)

    # Create TensorDataset for batches approach
    train_dataset    = TensorDataset(u_train, y_train)
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Log dataset dimension
    logger.info(f"Train dataset has size {u_train.size()}")
    logger.info(f"Validation dataset has size {u_valid.size()}")
    logger.info(f"Test dataset has size {u_test.size()}")
    logger.info("")

    ## MACHINE LEARNING MODEL
    # Initialization of the model
    torch.manual_seed(123)
    model_c = MLP(N_INPUT=(args.input_2d+1), N_OUTPUT=N_OUTPUT, N_HIDDEN=N_HIDDEN, N_LAYERS=N_LAYERS).to(device)
    model_k = MLP(N_INPUT=(args.input_2d+1), N_OUTPUT=N_OUTPUT, N_HIDDEN=N_HIDDEN, N_LAYERS=N_LAYERS).to(device)

    # Inizialization of the optimizer
    if (OPTIMIZER_TYPE == "SDG"): # Stochastic Gradient Descend (SDG) optimizer    
        optimizer = torch.optim.SGD(
            list(model_c.parameters()) + list( model_k.parameters()), 
            lr=LEARNING_RATE, momentum=MOMENTUM,
        )
    elif (OPTIMIZER_TYPE == "ADAM"): # Adam optimizer
        optimizer = torch.optim.Adam(
            list(model_c.parameters()) + list( model_k.parameters()), 
            lr=LEARNING_RATE,
        )
    else:
        raise ValueError("Optimizer type not supported. Please verify")

    # Initialization of adaptive learning rate
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=GAMMA,
        last_epoch=-1,
    )

    # Define the loss: mean square error
    MSE = nn.MSELoss()
    
    ## PHYSICAL MODEL
    # Load model parameters, get system matrices
    optimized_params = src.utils.load_params(MODEL_PATH)
    G, H, C, D, c, k_s, k_us, m_s, m_us = physical_model(optimized_params, dt=Ts, device=device)

    # Inversion of the model
    D_star = torch.linalg.pinv(D) # Pseudoinverse (Moore-Penrose inverse)
    C_star = - D_star @ C
    H_star = H @ D_star
    G_star = G + H @ C_star

    # Since the step function will be used several times,
    # define it here to maximize the reusability of the code
    def model_step( u, y, return_forces=False, testing=False):
        nonlocal model_c, model_k
        
        N = len(u)
        # The input is 2-dimensional: [acc, vehicle_speed]
        # Split the input to individually control the two information
        u_acc, u_vel = torch.split(u, split_size_or_sections=1, dim=2)

        # 1.Pre-allocate state and estimated output
        x = torch.zeros([N, G_star.shape[0], 1]).to(device)

        # 2.Compute the state
        for k in range(N-1):
            x[k+1] = (G_star @ x[k] + H_star @ u_acc[k]) # TODO: add NN
        
        # 3.Generate input and normalize it
        input_c = (x[:, 0,] - x[:, 1]).unsqueeze(2)
        input_k = (x[:, 2,] - x[:, 3]).unsqueeze(2)
        input_c_norm = min_max_normalization(input_c)
        input_k_norm = min_max_normalization(input_k)

         # 4.Feed the input (state differences) to the NN and get estimated forces
        if (args.input_2d): 
            u_vel_norm   = min_max_normalization(u_vel)
            input_c_norm = torch.cat((input_c_norm, u_vel_norm), dim=2)
            input_k_norm = torch.cat((input_k_norm, u_vel_norm), dim=2)
        
        if (testing):
            if (args.input_2d):
                offset_input = torch.cat((torch.zeros_like(u_vel_norm, device=device), u_vel_norm), dim=2)
            else:
                offset_input = torch.zeros_like(input_c_norm, device=device)
            
            offset_c = model_c(offset_input)
            offset_k = model_k(offset_input)
        else:
            offset_c, offset_k = 0, 0

        F_c = model_c(input_c_norm) - offset_c # f_c
        F_k = model_k(input_k_norm) - offset_k # f_k

        # 5.Estimate track vertical profile
        y_hat_linear    = (m_us*u_acc - k_s*(input_k) - c*(input_c) + k_us*(x[:, 3].unsqueeze(2)))/k_us 
        y_hat_nonlinear = -1*((F_c + F_k)/k_us)
        y_hat           = y_hat_linear + y_hat_nonlinear

        # Compute the error
        err = (y - y_hat)

        if (return_forces):
            return y_hat, y_hat_linear, y_hat_nonlinear, err, F_c, F_k
        else:
            return y_hat, y_hat_linear, y_hat_nonlinear, err, _, _

    def export_plots(data: tuple, description: str):
        nonlocal model_c, model_k

        # Unpack data
        u_export, y_export, track_export = data

        # Set to eval mode
        model_c.eval()
        model_k.eval()
        
        with torch.no_grad():
            y_hat, y_hat_linear, y_hat_nonlinear, err, F_c, F_k = model_step(u=u_export, y=y_export, return_forces=True, testing=True)
                                                                        
        d0 = PlotData(x=track_export, y=y_export,        label="Reference", linestyle="solid", alpha=0.4)
        d1 = PlotData(x=track_export, y=y_hat_linear,    label="Linear model", linestyle="solid", alpha=0.6)
        d2 = PlotData(x=track_export, y=y_hat,           label="Enhanced model", linestyle="solid", alpha=0.6)
        d3 = PlotData(x=track_export, y=y_hat_nonlinear, label="Non-linear contribution", linestyle="solid", alpha=0.6)
        d4 = PlotData(x=track_export, y=err,             label="Error", linestyle="solid", alpha=0.6, color="Red")  
        f1 = PlotData(x=track_export, y=F_c,             label="Fc", linestyle="solid", alpha=0.6)
        f2 = PlotData(x=track_export, y=F_k,             label="Fk", linestyle="solid", alpha=0.6)

        # 01-Reference vs. model vs. nonlinear model
        cool_plot(d0, d1, d2, ylim=[-0.0025, 0.0025], save=working_dir, filename=f'01_'+description, title=f"{args.job_id}: {description}") 
        cool_plot(d0, d1, d2, ylim=[-0.0025, 0.0025], xlim=[465, 485], save=working_dir, filename=f'narrow_01_'+description, title=f"{args.job_id}: {description}") 

        # 02-Reference vs. model vs. nonlinear model + forces contribution
        cool_plot(d0, d1, d2, d3, ylim=[-0.0025, 0.0025], save=working_dir, filename=f'02_'+description, title=f"{args.job_id}: {description}") 
        cool_plot(d0, d1, d2, d3, ylim=[-0.0025, 0.0025], xlim=[465, 485], save=working_dir, filename=f'narrow_02_'+description, title=f"{args.job_id}: {description}")  

        # 03-Reference vs. nonlinear model + error
        cool_plot(d0, d2, d4, ylim=[-0.0025, 0.0025], save=working_dir, filename=f'03_'+description, title=f"{args.job_id}: {description}") 
        cool_plot(d0, d2, d4, ylim=[-0.0025, 0.0025], xlim=[465, 485], save=working_dir, filename=f'narrow_03_'+description, title=f"{args.job_id}: {description}")

        # Plot estimated forces
        cool_plot(f1, f2, save=working_dir, filename=f'forces_'+description, title=f"{args.job_id}: Estimated forces") 

        # Plot NN contribution
        cool_plot(d3, save=working_dir, filename=f'NN_contribution_'+description, title=f"{args.job_id}: Neural network contribution [m]") 


        model_c.train()
        model_k.train()

    # NOTE: START OF TRAINING LOOP
    # Set the model to train mode
    counter = 0
    model_c.train()
    model_k.train()
    for epoch in range(NUM_EPOCHS):
        for batch_idx, (batch_u, batch_y) in enumerate(dataloader):
            
            # 0.Reset the gradient
            optimizer.zero_grad()

            # 1.Feed the input to the network and get the estimated output
            y_hat, y_hat_linear, y_hat_nonlinear, err, _, _= model_step(u=batch_u, y=batch_y)
           
            # 2.Define the loss as sum of different components
            # 2.1.Mean square error between estimated output and reference
            # loss1 = MSE(batch_y*1e3, y_hat*1e3)
            loss1 = torch.mean(((batch_y - y_hat)*1e3)**2)
            
            # 2.2.Enforce the contribution of the NN to be less than 1 cm
            # upper_bound = torch.ones_like(y_hat_nonlinear)*1e-3
            # loss2 = torch.mean(torch.relu(((torch.abs(y_hat_nonlinear) - upper_bound)*1e3)**2)) # Convert to mm
            # loss2 = 0
            y_hat_limited = torch.clamp(y_hat_nonlinear, min=-1e-3, max=1e-3)
            loss2 = torch.mean(torch.nn.functional.mse_loss(y_hat_limited, torch.zeros_like(y_hat_limited)))

            # 2.3.Enforce zero crossing
            if (args.input_2d):
                zero_point = torch.cat((torch.zeros_like(batch_u[:, :, 1], device=device), batch_u[:, :, 1]), dim=1).unsqueeze(1)
            else:
                zero_point = torch.zeros(size=[len(batch_u), 1], device=device)
            # loss3 = torch.mean(torch.abs(model_c(zero_point))) + torch.mean(torch.abs(model_k(zero_point)))*1e3 # convert to mm
            # loss3 = torch.mean(torch.abs(model_c(zero_point)/1e5)) + torch.mean(torch.abs(model_k(zero_point)/1e5))*1e3 # convert to mm
            # loss3 = torch.mean((model_c(zero_point)/1e5)) + torch.mean((model_k(zero_point)/1e5)) 
            # loss3 = 0
            loss3 = torch.mean((model_c(zero_point)/1e5)**2) + torch.mean((model_k(zero_point)/1e5)**2)

            loss = 1*loss1 + 0.2*loss2 + 0.7*loss3
            
            # 3.Optimize via gradient backpropagation
            loss.backward()
            optimizer.step()
            
            # Save every 1000 steps
            if (counter) % 1000 == 0: 
                                
                logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS} | Batch {batch_idx}/{len(dataloader)} | loss={loss}")
                logger.info(f"Current learning rate: {scheduler.get_last_lr()}")
                logger.info("Loss components")
                logger.info(f"\tMSE: {loss1}")
                logger.info(f"\tUpper bound: {loss2}")
                logger.info(f"\tzero crossing: {loss3}")
                logger.info("")

                export_plots(data=(u_valid, y_valid, track_valid), description=f"{(epoch+1):06d}")
                
            counter += 1
        
        # 10.Modify the learning rate
        if (USE_SCHEDULER): scheduler.step()

    # Test set
    export_plots(data=(u_test, y_test, track_test), description=f"Test_dataset")

    # Export graphs as GIF
    pattern = ["\\01_", "\\02_", "\\03_", "\\forces_", "\\narrow_01_", "\\narrow_02_", "\\narrow_03_", ]
    save_as_gif(directory=working_dir, sub_folder="plots", pattern=pattern, logger=logger)
   
if __name__ == '__main__':
    global logger

    args = parser.parse_args()

    # Create optimization folder
    if (args.job_id == ""):
        folder = src.utils.create_dir(folder_base_name='./model_output', folder_sub_name='run')
    else:
        folder = src.utils.create_dir(folder_base_name='./model_output', job_id=f'MLP_{args.job_id}')

    # Initialize logger
    logger = src.utils.get_logger(logpath=os.path.join(folder, 'logs'), filepath=__file__, displaying=True, saving=True, )

    main(folder, args)
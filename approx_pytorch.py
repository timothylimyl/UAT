# Downloading dependencies:
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

### Setting function to approximate: *******************   #

# Actual Function here, Relationship: y = x^2
x = np.linspace(-30,30,100)
y = x**2

# Feel free to change Actual Function whatever u want to try: Ex: y=sin(x)
#x = np.linspace(0,180,200)
#y = np.sin(np.deg2rad(x))


#   ****************************************************   #

### Setting up the feedfoward neural network  **********   #

# Hyperparamters to tune: Number of Neurons and Hidden Layers, Learning Rate and Epochs.
n_neurons = 50  # number of neurons/nodes
learning_rate = 5e-3 # learning rate

   
model = nn.Sequential(     
          nn.Linear(1, n_neurons),
          nn.ReLU(),
          #nn.Linear(n_neurons,n_neurons),
          #nn.ReLU(),        
          nn.Linear(n_neurons,1),
          nn.ReLU()
          )


# Set up  : Input (1 Node) -> Hidden (10 nodes) -> Output (1 Node) 
# Set up 2: Input (1 Node) -> Hidden (10 nodes) -> Hidden (10 nodes) -> Output (1 Node)

# Important Note: If you increase the number of neurons or use a harder function to approximate, try tuning the learning rate.
#                 Tuning the learning rate is vital to properly train the network.

optimizer = optim.RMSprop(model.parameters(), lr=learning_rate) # define optimizer
#optimizer = optim.SGD(model.parameters(), lr=learning_rate)

criterion = nn.MSELoss() # define loss function

# ******************************************************  #

### Training: ******************************************  #

# Convert to tensor form with batch for PyTorch model.
inputs = torch.tensor(x).view(-1,1)
labels = torch.tensor(y).view(-1,1)

# Important Note 2: Change epochs
epochs = 20000
for epoch in range(epochs):  # loop over the data multiple times
   
    # zero the parameter gradients
    optimizer.zero_grad()
   
    # forward + backward + optimize
    outputs = model(inputs.float())
    loss = criterion(outputs, labels.float())
    loss.backward()
    optimizer.step()
    
# ******************************************************  #

### Running Inference over the trained model ***********  #
with torch.no_grad():
    test_inputs = torch.tensor(x).view(len(x),-1).float()
    y_hat = model(test_inputs)
    y_hat = y_hat.detach().numpy()
    
# ******************************************************  #

### Plot results: Actual vs Model Prediction ***********  #
plt.scatter(x,y,label='Actual Function')
plt.scatter(x,y_hat,label="Predicted Function")
plt.title(f'Number of neurons: {n_neurons}')
plt.xlabel('Input Variable (x)')
plt.ylabel('Output Variable (y)')
plt.legend()
plt.show()
# ******************************************************  #


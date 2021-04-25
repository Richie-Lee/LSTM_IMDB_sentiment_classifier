"""
Assignment 4 code - Richie Lee (KLE680)

This code evaluates the performance in terms of LOSS and ACCURACY

brief code structure:
    - Data Loading (IMDb)
    - Variable Batch Size implementation
    - MLP
    - Elman (Self-build)
    - Elman (Torch Built-in)
    - LSTM  (Torch Built-in)
    - Training function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import wget, os, gzip, pickle, random

from datetime import datetime



# Define hyperparameter & additional info
Learn_rate = 0.01
Epochs = 1

# Execution time
startTime = datetime.now()



# Data Loader (IMDb)
"""
Downloads IMDb data from github
Data: 50.000 reviews

Data splits
- final = True: [Train / Test] = [25.000 / 25.000]
- final = False [Train / Validation] = [20.000 / 5.000]

"""

IMDB_URL = 'http://dlvu.github.io/data/imdb.{}.pkl.gz'
IMDB_FILE = 'imdb.word.pkl.gz'

# Data Loader (final = True [test/train] & final = False [validation/train])
def load_imdb(final=True, val=5000, seed=0, voc=None, char=False):

    cst = 'char' if char else 'word'

    imdb_url = IMDB_URL.format(cst)
    imdb_file = IMDB_FILE.format(cst)

    if not os.path.exists(imdb_file):
        wget.download(imdb_url)

    with gzip.open(imdb_file) as file:
        sequences, labels, i2w, w2i = pickle.load(file)

    if voc is not None and voc < len(i2w):
        nw_sequences = {}

        i2w = i2w[:voc]
        w2i = {w: i for i, w in enumerate(i2w)}

        mx, unk = voc, w2i['.unk']
        for key, seqs in sequences.items():
            nw_sequences[key] = []
            for seq in seqs:
                seq = [s if s < mx else unk for s in seq]
                nw_sequences[key].append(seq)

        sequences = nw_sequences

    if final:
        return (sequences['train'], labels['train']), (sequences['test'], labels['test']), (i2w, w2i), 2

    # Make a validation split
    random.seed(seed)

    x_train, y_train = [], []
    x_val, y_val = [], []

    val_ind = set( random.sample(range(len(sequences['train'])), k=val) )
    for i, (s, l) in enumerate(zip(sequences['train'], labels['train'])):
        if i in val_ind:
            x_val.append(s)
            y_val.append(l)
        else:
            x_train.append(s)
            y_train.append(l)

    return (x_train, y_train), \
           (x_val, y_val), \
           (i2w, w2i), 2

(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final = True)



# Variable Batch Size 
"""
Slices data into batches with approximately equal amounts of tokens
Pads accordingly
"""

# Padding function
def pad(l, size, padding):
    return l + [padding] * abs((len(l)-size))

# Variable batch slicing
def batch_slicing(x, y):
    
    # Initialize variables (e.g. x_B = x_batched)
    batch, x_B, y_B = [], [], []
    current_batch, current_size = 0, 0
    i = 1
    
    # Maximum # Tokens in batch = # Tokens in largest sequence (x_train = sorted)
    max_tokens = len(x[-1])
    
    # Iterates over dataset and slices into appropiate sizes (token-based)
    while i != len(x) + 1:
        batch.append(x[i - 1])
        current_size += len(x[i - 1])
        
        # Final batch/element (longest sequence)
        if i == len(x):
            x_B.append(batch)
            y_B.append(y[-1:])
        
        # Slice batch if max_tokens condition violated
        elif current_size + len(x[i]) >= max_tokens:
            
            # Padding for batch
            for j in range(len(batch)):
                batch[j] = pad(batch[j], size = len(batch[-1]), padding = 0)
            
            # store current batch
            x_B.append(batch)
            y_B.append(y[i - len(batch): i])
            
            # reset variables
            current_size = 0
            batch = []
            current_batch += 1
            
        i += 1
    
    # convert to Torch Tensors
    for i in range(len(x_B)):
        x_B[i] = torch.tensor(x_B[i], dtype = torch.long)
        y_B[i] = torch.tensor(y_B[i], dtype = torch.long)

    return x_B, y_B

# Obtained batched training data
x_train_B, y_train_B = batch_slicing(x_train, y_train)
x_val_B, y_val_B = batch_slicing(x_val, y_val)



# MLP
"""
MLP network Class 
utilizes PyTorch built-in functions
"""

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.emb = nn.Embedding(num_embeddings = len(i2w), embedding_dim = 300)
        self.fc1 = nn.Linear(300, 300)
        self.fc2 = nn.Linear(300, 2)
    
    def forward(self, x):
        x = self.emb(x)
        x = self.fc1(x)
        x = F.relu(x)
        x, y = torch.max(x, dim = 1) # global max pool
        x = self.fc2(x)
        
        return x    

mlp_net = MLP()



# Elman (without Torch built-in function)
"""
Elman Network Class, and Elman (selfbuild) layer class  
doesn't utilize PyTorch built-in RNN function, uses "Elman_selfbuild" class instead
"""

class Elman(nn.Module):
    def __init__(self, insize=300, outsize=300, hsize=300):
        super().__init__()
        self.fc1 = nn.Linear(insize + hsize, hsize)
        self.fc2 = nn.Linear(hsize, outsize)
    
    def forward(self, x, hidden=None):
        b, t, e = x.size()
        
        # first iteration
        if hidden is None:
            hidden = torch.zeros(b, e, dtype = torch.float)
            
        outputs_list = []
        for i in range(t):
            inputs = torch.cat([x[:, i, :], hidden], dim = 1)
            hidden = torch.sigmoid(self.fc1(inputs))
            outputs = self.fc2(hidden)
            outputs_list.append(outputs[:, None, :])
            
        return torch.cat(outputs_list, dim = 1), hidden

class Elman_selfbuild(nn.Module):
    def __init__(self):
        super(Elman_selfbuild, self).__init__()
        self.emb = nn.Embedding(num_embeddings = len(i2w), embedding_dim = 300)
        self.elman = Elman()
        self.fc2 = nn.Linear(300, 2)
    
    def forward(self, x):
        x = self.emb(x)
        x_residual, hidden = self.elman(x) # residual connection 
        x, y = torch.max(x, dim = 1) # global max pool
        x = self.fc2(x)
        
        return x    

elman_selfbuild_net = Elman_selfbuild()



# Elman (with Torch)
"""
Elman Network Class
utilizes PyTorch built-in function
"""

class Elman_Torch(nn.Module):
    def __init__(self):
        super(Elman_Torch, self).__init__()
        self.emb = nn.Embedding(num_embeddings = len(i2w), embedding_dim = 300)
        self.elman = torch.nn.RNN(input_size=300, hidden_size=300) # tanh non-linearity
        self.fc2 = nn.Linear(300, 2)
     
    def forward(self, x):
        x = self.emb(x)
        x_residual, hidden = self.elman(x) # residual connection 
        x = x + x_residual        
        x, y = torch.max(x, dim = 1) # global max pool
        x = self.fc2(x)
        
        return x    

elman_torch_net = Elman_Torch()



# LSTM (with Torch)
"""
LSTM Network Class
utilizes PyTorch built-in function
"""
class LSTM_Torch(nn.Module):
    def __init__(self):
        super(LSTM_Torch, self).__init__()
        self.emb = nn.Embedding(num_embeddings = len(i2w), embedding_dim = 300)
        self.lstm = torch.nn.LSTM(input_size=300, hidden_size=300)
        self.fc2 = nn.Linear(300, 2)
     
    def forward(self, x):
        x = self.emb(x)
        x, hidden = self.lstm(x)
        x, y = torch.max(x, dim = 1) # global max pool
        x = self.fc2(x)
        
        return x    

lstm_torch_net = LSTM_Torch()



# Training Function
"""
Training + Validation/Test function. 
Model should still be specified in optimizer and in outputs (in function)
"""

# Defining the Optimizer and Loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lstm_torch_net.parameters(), lr = Learn_rate)

# Training Function
def train(Learn_rate, Epochs):
    
    # initialise lists for plots
    train_loss_list = []
    train_accuracy_list = []
    
    for epoch in range(Epochs):
        print(f"\nEpoch {epoch+1}/{Epochs}:")
        
        # initialization
        running_loss = 0
        running_loss_total = 0
        correct = 0
        seq_count = 0
        
        # Training
        for i in range(len(x_train_B)):
            
            # counting running total amount of sequences (for accuracy evaluation)
            seq_count += len(x_train_B[i])
            
            # get inputs and labels
            inputs, labels = x_train_B[i], y_train_B[i]
        
            # zero the parameter gradients
            optimizer.zero_grad()
        
            # forward + backward + optimize
            outputs = lstm_torch_net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            # update 
            correct += sum(labels == torch.argmax(outputs, dim = 1))
            running_loss += loss.item()
            running_loss_total += loss.item()

            # Print progress
            if i % 10 == 9:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0
                # running average loss/accuracy (up to iteration i)
                train_loss_list.append(running_loss_total / i)
                train_accuracy_list.append((correct/seq_count).item())
        
        # Computing & printing epoch's loss/accuracy (train)
        train_loss = running_loss_total / len(x_train_B)
        train_accuracy = (correct/seq_count).item()
        print(f"\ntrain loss {epoch+1}:     = {train_loss}")
        print(f"train accuracy {epoch+1}:   = {train_accuracy}\n")
        
        
        # Validation (or test)
        running_loss_total = 0
        correct = 0
        seq_count = 0
        for i in range(len(x_val_B)):
            
            # counting running total amount of sequences (for accuracy evaluation)
            seq_count += len(x_val_B[i])
            
            # get inputs and labels
            inputs, labels = x_val_B[i], y_val_B[i]
            
            # forward + backward + optimize
            outputs = lstm_torch_net(inputs)
            loss = criterion(outputs, labels)
            
            # update metrics
            correct += sum(labels == torch.argmax(outputs, dim = 1))
            running_loss_total += loss.item()
         
        # Computing & printing epoch's loss/accuracy (train)
        val_loss = running_loss_total / len(x_val_B)
        val_accuracy = (correct/seq_count).item()
        print(f"validation loss {epoch+1}:      = {val_loss}")
        print(f"validation accuracy {epoch+1}:  = {val_accuracy}\n")
        
    return train_loss_list, train_accuracy_list



# Execution 
"""
Executes model and prints final relevant info
"""

# Start training + test/validation
train_loss_list, train_accuracy_list = train(Learn_rate, Epochs)
print(f"\nTRAIN LOSS:\n{train_loss_list}")
print(f"\nTRAIN ACCURACY:\n{train_accuracy_list}")

# Print exectution time
print(f"\ntotal runtime: {datetime.now() - startTime}")


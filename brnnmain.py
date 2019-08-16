from Pytorch_BRNN import *
from torch.utils.data import DataLoader,random_split
from torch.utils.data.sampler import SubsetRandomSampler
from pytorchtools import EarlyStopping
from tqdm import tqdm
import argparse
import os
import torch.optim as optim
try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")

parser = argparse.ArgumentParser(description = "Determine the Type of Cells and Loss Function to be Used")
parser.add_argument('-clstm','--convlstm', help = 'use convlstm as base cell', action = 'store_true')
parser.add_argument('-cgru','--convgru', help = 'use convgru as base cell', action = 'store_true')
parser.add_argument('-MSE', '--MSELoss', help = 'use MSE as loss function', action = 'store_true')
parser.add_argument('-xentro', '--crossentropyloss', help = 'use Cross Entropy Loss as loss function', action = 'store_true')
args = parser.parse_args()

if args.convlstm:
    basecell = 'CLSTM'
if args.convgru:
    basecell = 'CGRU'
else:
    basecell = 'CGRU'

if args.MSELoss:
    objectfunction = 'MSELoss'
if args.crossentropyloss:
    objectfunction = 'crossentropyloss'
else:
    objectfunction = 'MSELoss'


###Dataset and Dataloader
batch_size = 8
# percentage of training set to use as validation
valid_size = 0.2
shuffle_dataset = True
random_seed= 1996
n_epochs = 500
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.device_count()>1:
    torch.cuda.manual_seed_all(random_seed)
else:
    torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

mnistdata = MovingMNISTdataset("data/mnist_test_seq.npy")
train_size = int(0.8 * len(mnistdata))
test_size = len(mnistdata) - train_size
torch.manual_seed(torch.initial_seed())
train_dataset, test_dataset = random_split(mnistdata, [train_size, test_size])

num_train = len(train_dataset)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_idx, valid_idx = indices[split:], indices[:split]
    
# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

 # load training data in batches
train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          sampler=train_sampler)
    
# load validation data in batches
valid_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          sampler=valid_sampler)
    
# load test data in batches
test_loader = DataLoader(test_dataset,
                         batch_size=batch_size)

CRNN_num_features=[64,32,32]
CRNN_filter_size=5
CRNN_shape=(64,64)#H,W
CRNN_inp_chans=1

CRNNargs = [CRNN_shape, CRNN_inp_chans, CRNN_filter_size, CRNN_num_features]

decoder_shape = (64, 64)
decoder_input_channels = 2*sum(CRNN_num_features)
decoder_filter_size = 1
decoder_num_features = 1

decoderargs = [decoder_shape, decoder_input_channels, decoder_filter_size, decoder_num_features]

def train():
    '''
    main function to run the training
    '''
    net = PredModel(CRNNargs, decoderargs, cell = basecell)
    tb = SummaryWriter()
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=10, verbose=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count()>1:
        net = nn.DataParallel(net)
        multipleDevice = True
    else:
        multipleDevice = False

    net.to(device)

    if objectfunction == 'MSELoss':
        lossfunction = nn.MSELoss().cuda()
    
    optimizer = optim.RMSprop(net.parameters(), lr = 0.0001)

    if multipleDevice:
        hidden_state = net.module.init_hidden(batch_size)
    else:
        hidden_state = net.init_hidden(batch_size)
    
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    for epoch in range(1, n_epochs + 1):
        ###################
        # train the model #
        ###################
        t = tqdm(train_loader,leave=False,total=len(train_loader))
        for data in t:
            input = data[:, 0:10, ...].to(device)
            label = data[:, 10:20, ...].to(device)
            
            optimizer.zero_grad()

            if multipleDevice:
                hidden_state = net.module.init_hidden(batch_size)
            else:
                hidden_state = net.init_hidden(batch_size)

            pred = net(input, hidden_state)

            if objectfunction == 'MSELoss':
                loss = 0
                for seq in range(10):
                    labelframe = label[:, seq, ...].view(batch_size, -1)
                    predframe = pred[seq].view(batch_size, -1)
                    curloss = lossfunction(predframe, labelframe)
                    loss += curloss

            if objectfunction == 'crossentropyloss':
                loss= 0
                for seq in range(10):
                    predframe = torch.sigmoid(pred[seq].view(batch_size, -1))
                    labelframe = label[:, seq, ...].view(batch_size, -1)
                    curloss = crossentropyloss(predframe, labelframe)
                    loss += curloss  
            loss_aver = loss.item() / batch_size                              
            loss.backward()
            optimizer.step()          
            # print ("trainloss: {:.6f},  epoch : {:02d}".format(loss_aver,epoch),end = '\r', flush=True)
            t.set_postfix({'trainloss': '{:.6f}'.format(loss_aver),'epoch' : '{:02d}'.format(epoch)})  
            train_losses.append(loss_aver)
        tb.add_scalar('TrainLoss',loss_aver,epoch)
        ######################    
        # validate the model #
        ######################  
        with torch.no_grad():         
            for data in valid_loader:
                input = data[:, 0:10, ...].to(device)
                label = data[:, 10:20, ...].to(device)

                if multipleDevice:
                    hidden_state = net.module.init_hidden(batch_size)
                else:
                    hidden_state = net.init_hidden(batch_size)

                pred = net(input, hidden_state)

                if objectfunction == 'MSELoss':
                    loss = 0
                    for seq in range(10):
                        labelframe = label[:, seq, ...].view(batch_size, -1)
                        predframe = pred[seq].view(batch_size, -1)
                        curloss = lossfunction(predframe, labelframe)
                        loss += curloss
        
                if objectfunction == 'crossentropyloss':
                    loss = 0
                    for seq in range(10):
                        predframe = torch.sigmoid(pred[seq].view(batch_size, -1))
                        labelframe = label[:, seq, ...].view(batch_size, -1)
                        curloss = crossentropyloss(predframe, labelframe)
                        loss += curloss
                loss_aver = loss.item() / batch_size 
                # print ("validloss: {:.6f},  epoch : {:02d}".format(loss_aver,epoch),end = '\r', flush=True)
                # record validation loss
                valid_losses.append(loss_aver)
                t.set_postfix({'validloss': '{:.6f}'.format(loss_aver),'epoch' : '{:02d}'.format(epoch)}) 

        tb.add_scalar('ValidLoss',loss_aver,epoch)
        torch.cuda.empty_cache()
        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        epoch_len = len(str(n_epochs))
        
        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.6f} ' +
                     f'valid_loss: {valid_loss:.6f}')
        
        print(print_msg)
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, net)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

    with open("avg_train_losses.txt",'wt') as f:
        for i in avg_train_losses:
            print(i, file=f)

    with open("avg_valid_losses.txt",'wt') as f:
        for i in avg_valid_losses:
            print(i, file=f)
    

def test():
    file_path = os.getcwd() + '/trained_model'
    testnet = torch.load(file_path)

def inference():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count()>1:
        #net = nn.DataParallel(net)
        multipleDevice = True
    else:
        multipleDevice = False

    file_path = os.getcwd() + '/trained_model'
    inferencenet = torch.load(file_path)

    for data in test_loader:

        input = data[:, 0:10, ...].to(device)
        if multipleDevice:
            hidden_state = inferencenet.module.init_hidden(batch_size)
        else:
            hidden_state = inferencenet.init_hidden(batch_size)

        pred = inferencenet(input, hidden_state)
        pred_np = []
        for i in range(len(pred)):
            append = pred[i].cpu()
            pred_np.append(append.data.numpy())
        break

    np.save(os.getcwd()+'/input', input.cpu())
    np.save(os.getcwd()+'/label', data[:, 10:20, ...].cpu())
    np.save(os.getcwd()+'/inference', pred_np)

if __name__ == "__main__":
    train()
    inference()

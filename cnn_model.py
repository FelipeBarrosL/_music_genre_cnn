import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

ALLOWED_CLASSES = ['normal', 'murmur', 'extrahls', 'artifact']
NUM_CLASSES = len(ALLOWED_CLASSES)

def run():
    set_a_train_path = './Data/set_a_train'
    set_a_test_path = './Data/set_a_test'
    set_b_train_path = './Data/set_b_train'
    set_b_test_path = './Data/set_b_test'

    #Load train and test data with batch_size = 4
    trasnform = transforms.Compose(
        [transforms.Resize((100,100)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5))]
    )

    batch_size = 4

    train_set = torchvision.datasets.ImageFolder(root=set_a_train_path, transform=trasnform)
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=True)

    test_set = torchvision.datasets.ImageFolder(root=set_a_test_path, transform=trasnform)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size, shuffle=True)

    #Define the CNN layers and other funcions
    class ImageMulticlassCNNClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2,2)
            self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(100*100, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, NUM_CLASSES)
            self.relu = nn.ReLU()
            self.softmax = nn.LogSoftmax()

        def forward(self,x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.pool(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = self.pool(x)
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            x = self.softmax(x)

            return x

    #Creating instance of CNN model and defining the loss function
    model = ImageMulticlassCNNClassifier()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    #TRAINING 
    losses_epoch_mean = []

    n_epochs = 100

    for epoch in range(n_epochs):

        losees_epoch = []

        for i, data in enumerate(train_loader, 0):

            X, label = data

            optimizer.zero_grad()
            output = model(X)

            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            losees_epoch.append(loss.item())
        
        losses_epoch_mean.append(np.mean(losees_epoch))
        print(f"Epoch {epoch}   -> Loss: {np.mean(losees_epoch):.4f}")


    #Plot Losses
    losses_plot = sns.lineplot(x=list(range(len(losses_epoch_mean))),y=losses_epoch_mean)
    fig = losses_plot.get_figure()
    fig.savefig("losses_per_epoch.png") 

    #TESTING

    y_test = []
    y_test_pred = []

    for i, data in enumerate(test_loader, 0):
        
        X, label = data

        with torch.no_grad():
            y_test_pred_temp = model(X).round()
        
        y_test.extend(label.numpy())
        y_test_pred.extend(y_test_pred_temp.numpy())


    #Metrics for eval
    acc = accuracy_score(y_test, np.argmax(y_test_pred, axis=1))
    print(f"Accuracy: {acc*100:.3f}%")

    #Plotting confusion matrix
    cm = confusion_matrix(y_test, np.argmax(y_test_pred, axis=1))
    losses_plot = sns.heatmap(cm, annot=True, xticklabels=ALLOWED_CLASSES, yticklabels=ALLOWED_CLASSES)
    fig = losses_plot.get_figure()
    fig.savefig("confusion_matrix.png") 


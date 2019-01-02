
# coding: utf-8

# # DATA20001 Deep Learning - Group Project
# ## Image project
# 
# **Due Thursday, December 13, before 23:59.**
# 
# The task is to learn to assign the correct labels to a set of images.  The images are originally from a photo-sharing site and released under Creative Commons-licenses allowing sharing.  The training set contains 20 000 images. We have resized them and cropped them to 128x128 to make the task a bit more manageable.
# 
# We're only giving you the code for downloading the data. The rest you'll have to do yourselves.
# 
# Some comments and hints particular to the image project:
# 
# - One image may belong to many classes in this problem, i.e., it's a multi-label classification problem. In fact there are images that don't belong to any of our classes, and you should also be able to handle these correctly. Pay careful attention to how you design the outputs of the network (e.g., what activation to use) and what loss function should be used.
# 
# - As the dataset is pretty imbalanced, don't focus too strictly on the outputs being probabilistic. (Meaning that the right threshold for selecting the label might not be 0.5.)
# 
# - Image files can be loaded as numpy matrices for example using `imread` from `matplotlib.pyplot`. Most images are color, but a few grayscale. You need to handle the grayscale ones somehow as they would have a different number of color channels (depth) than the color ones.
# 
# - In the exercises we used e.g., `torchvision.datasets.MNIST` to handle the loading of the data in suitable batches. Here, you need to handle the dataloading yourself.  The easiest way is probably to create a custom `Dataset`. [See for example here for a tutorial](https://github.com/utkuozbulak/pytorch-custom-dataset-examples).

# ## Download the data

# In[1]:

import os
import torch
import torchvision
from torchvision.datasets.utils import download_url
import zipfile

train_path = 'train'
dl_file = 'dl2018-image-proj.zip'
dl_url = 'https://users.aalto.fi/mvsjober/misc/'

zip_path = os.path.join(train_path, dl_file)
if not os.path.isfile(zip_path):
    download_url(dl_url + dl_file, root=train_path, filename=dl_file, md5=None)

with zipfile.ZipFile(zip_path) as zip_f:
    zip_f.extractall(train_path)
    #os.unlink(zip_path)

print("Files downloaded!")


# The above command downloaded and extracted the data files into the `train` subdirectory.
# 
# The images can be found in `train/images`, and are named as `im1.jpg`, `im2.jpg` and so on until `im20000.jpg`.
# 
# The class labels, or annotations, can be found in `train/annotations` as `CLASSNAME.txt`, where CLASSNAME is one of the fourteen classes: *baby, bird, car, clouds, dog, female, flower, male, night, people, portrait, river, sea,* and *tree*.
# 
# Each annotation file is a simple text file that lists the images that depict that class, one per line. The images are listed with their number, not the full filename. For example `5969` refers to the image `im5969.jpg`.

# ## Your stuff goes here ...

# In[61]:

# Here goes your code ...
# get_ipython().magic('matplotlib inline')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from PIL import Image, ImageOps

height = 224
width = 224
num_images = 20000

data = torch.empty(num_images,height,width,3,dtype=torch.float)

print("Loading data...")
for i in range(num_images):
    path = "train/images/im"+str(i+1)+".jpg"
    element = np.asarray(Image.open(path).resize((height,width), Image.ANTIALIAS))
    element = element/np.max(element)
    element = torch.from_numpy(element)
    #element = torch.from_numpy(np.asarray(Image.open(path).resize((height,width), Image.ANTIALIAS)))
    if element.shape != torch.Size([height, width,3]):    #Duplicate grayscale images to all of the channels
        element = torch.stack([element,element,element],2)
    data[i] = element
print("Data loading done!")


# In[62]:

classes = {
    "baby": 0,
    "bird": 1,
    "car": 2,
    "clouds": 3,
    "dog": 4,
    "female": 5,
    "flower": 6,
    "male": 7,
    "night": 8,
    "people": 9,
    "portrait": 10,
    "river": 11,
    "sea": 12,
    "tree": 13
}

# Open each of the files containing the information about which images belong to which labels
# and create a set for each label containing the numbers of the images in belonging to that label

with open("train/annotations/baby.txt") as file:
    
    lines = file.read()
    lines = lines.split("\n")
    baby = set([])
    for line in lines:
        if line!="\n" and line != '':  # ignore newlines and empty lines
            baby.add(int(line))

with open("train/annotations/bird.txt") as file:
    
    lines = file.read()
    lines = lines.split("\n")
    bird = set([])
    for line in lines:
        if line!="\n" and line != '':
            bird.add(int(line))

with open("train/annotations/car.txt") as file:
    
    lines = file.read()
    lines = lines.split("\n")
    car = set([])
    for line in lines:
        if line!="\n" and line != '':
            car.add(int(line))

with open("train/annotations/clouds.txt") as file:
    
    lines = file.read()
    lines = lines.split("\n")
    clouds = set([])
    for line in lines:
        if line!="\n" and line != '':
            clouds.add(int(line))

with open("train/annotations/dog.txt") as file:
    
    lines = file.read()
    lines = lines.split("\n")
    dog = set([])
    for line in lines:
        if line!="\n" and line != '':
            dog.add(int(line))
        
with open("train/annotations/female.txt") as file:
    
    lines = file.read()
    lines = lines.split("\n")
    female = set([])
    for line in lines:
        if line!="\n" and line != '':
            female.add(int(line))
        
with open("train/annotations/flower.txt") as file:
    
    lines = file.read()
    lines = lines.split("\n")
    flower = set([])
    for line in lines:
        if line!="\n" and line != '':
            flower.add(int(line))
        
with open("train/annotations/male.txt") as file:
    
    lines = file.read()
    lines = lines.split("\n")
    male = set([])
    for line in lines:
        if line!="\n" and line != '':
            male.add(int(line))
        
with open("train/annotations/night.txt") as file:
    
    lines = file.read()
    lines = lines.split("\n")
    night = set([])
    for line in lines:
        if line!="\n" and line != '':
            night.add(int(line))

with open("train/annotations/people.txt") as file:
    
    lines = file.read()
    lines = lines.split("\n")
    people = set([])
    for line in lines:
        if line!="\n" and line != '':
            people.add(int(line))
        
with open("train/annotations/portrait.txt") as file:
    
    lines = file.read()
    lines = lines.split("\n")
    portrait = set([])
    for line in lines:
        if line!="\n" and line != '':
            portrait.add(int(line))
        
with open("train/annotations/river.txt") as file:
    
    lines = file.read()
    lines = lines.split("\n")
    river = set([])
    for line in lines:
        if line!="\n" and line != '':
            river.add(int(line))
        
with open("train/annotations/sea.txt") as file:
    
    lines = file.read()
    lines = lines.split("\n")
    sea = set([])
    for line in lines:
        if line!="\n" and line != '':
            sea.add(int(line))
        
with open("train/annotations/tree.txt") as file:
    
    lines = file.read()
    lines = lines.split("\n")
    tree = set([])
    for line in lines:
        if line!="\n" and line != '':
            tree.add(int(line))
            
class_list = [baby,bird,car,clouds,dog,female,flower,male,night,people,portrait,river,sea,tree]

def find_classes(index):
    classes = np.zeros(len(class_list))
    for i in range(len(class_list)):
        if index in class_list[i]:  # find out if image number belongs to the label
            classes[i]=1            # and set that labels array element to 1
    # returns an array indicating to which labels the image belongs to (1 = belongs to label, 0 = doesn't belong)
    return classes
print("List of classes formed!")


# In[63]:

classes = np.zeros((len(data),len(class_list)))

#create list containing all of the class labels for all of the images
for i in range(len(data)):
    classes[i]=find_classes(i+1)
    
classes = torch.from_numpy(classes)

print("Classes assinged!")



# In[64]:

from torch.utils.data.dataset import Dataset
from torchvision import transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])

class CustomDataset(Dataset):
    def __init__(self, data, labels, transforms=None):
        self.data = data
        self.labels = labels
        self.height = data.shape[1]
        self.width = data.shape[2]
        self.transforms = transforms

    def __getitem__(self, index):
        image_label = self.labels[index]
        image = self.data[index]
        if self.transforms is not None:
            image = self.transforms(image)
        return (image, image_label)

    def __len__(self):
        return len(self.data)

#create a dataset from the data loaded before and the corresponding class labels
training_data = data[0:16000]
training_classes = classes[0:16000]
validation_data = data[16000:len(data)]
validation_classes = classes[16000:len(data)]
training_set = CustomDataset(training_data.transpose(3,1).transpose(2,3),training_classes,transforms = normalize)
validation_set = CustomDataset(validation_data.transpose(3,1).transpose(2,3),validation_classes,transforms = normalize)
print("Datasets created!")


# In[65]:

train_loader = torch.utils.data.DataLoader(dataset=training_set,
                                                    batch_size=1,
                                                    shuffle=False)

validation_loader = torch.utils.data.DataLoader(dataset=validation_set,
                                                    batch_size=1,
                                                    shuffle=False)


print("Dataloaders created!")


# In[66]:

from torchvision import models

resnet = models.resnet18()
state_dict = torch.utils.model_zoo.load_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth','/wrk/mnoora')
resnet.load_state_dict(state_dict)

#print(resnet)


# In[67]:
#Freeze parameters
for parameter in resnet.parameters():
    parameter.requires_grad = False

def get_trainable_parameters(model):
    return (parameter for parameter in model.parameters() if parameter.requires_grad)

#Replace the linear layer with a new one
resnet.fc = nn.Linear(in_features=512, out_features=classes.shape[1], bias=True)
model = nn.Sequential(resnet,nn.Sigmoid())
if torch.cuda.is_available():
    print('Using GPU!')
    device = torch.device('cuda')
else:
    print('Using CPU')
    device = torch.device('cpu')
model = model.to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(get_trainable_parameters(model), lr =0.01)


# In[68]:

def train(epoch, log_interval=100):
        model.train()
        # Loop over each batch from the training set
        for batch_idx, (data, target) in enumerate(train_loader):
            # Copy data to GPU if needed
            data = data.to(device)
            target = target.to(device)
            # Zero gradient buffers
            optimizer.zero_grad()
            # Pass data through the network
            output = model(data)
            # Calculate loss
            
            loss = criterion(output, target.float())
            # Backpropagate
            loss.backward()
            # Update weights
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data.item()))
            
def validate(loss_vector, accuracy_vector):
        model.eval()
        val_loss, correct = 0, 0
        for data, target in validation_loader:
            data = data.to(device)
            target = target.float()
            target = target.to(device)
            output = model(data)
            val_loss += criterion(output, target).data.item()
            pred = output.data.round().float()
            correct += pred.eq(target.data).cpu().sum().item()==14

        val_loss /= len(validation_loader)
        loss_vector.append(val_loss)
        accuracy = 100. * correct / len(validation_loader.dataset)
        accuracy_vector.append(accuracy)
        with open("train/accuracy.txt","a") as file:
            text = "Accuracy: "+str(correct)+"/"+str(len(validation_loader.dataset))+ str(accuracy)+"\n"
            file.write(text)
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            val_loss, correct, len(validation_loader.dataset), accuracy))


# In[69]:

epochs = 20
lossv, accv = [], []
for epoch in range(1, epochs + 1):
    train(epoch)
    validate(lossv, accv)


# In[48]:




# ## Save your model
# 
# It might be useful to save your model if you want to continue your work later, or use it for inference later.

# In[34]:

torch.save(model.state_dict(), 'model.pkl')


# The model file should now be visible in the "Home" screen of the jupyter notebooks interface.  There you should be able to select it and press "download".  [See more here on how to load the model back](https://github.com/pytorch/pytorch/blob/761d6799beb3afa03657a71776412a2171ee7533/docs/source/notes/serialization.rst) if you want to continue training later.

# ## Download test set
# 
# The testset will be made available during the last week before the deadline and can be downloaded in the same way as the training set.

# ## Predict for test set
# 
# You should return your predictions for the test set in a plain text file.  The text file contains one row for each test set image.  Each row contains a binary prediction for each label (separated by a single space), 1 if it's present in the image, and 0 if not. The order of the labels is as follows (alphabetic order of the label names):
# 
#     baby bird car clouds dog female flower male night people portrait river sea tree
# 
# An example row could like like this if your system predicts the presense of a bird and clouds:
# 
#     0 1 0 1 0 0 0 0 0 0 0 0 0 0
#     
# The order of the rows should be according to the numeric order of the image numbers.  In the test set, this means that the first row refers to image `im20001.jpg`, the second to `im20002.jpg`, and so on.

# If you have the prediction output matrix prepared in `y` you can use the following function to save it to a text file.

# In[ ]:

#np.savetxt('results.txt', y, fmt='%d')


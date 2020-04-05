import sys
import torch
import torch.utils.data as data
import torch.nn as nn
from torchvision import models

import numpy as np
from PIL import Image
import glob
import random
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2
import pickle

device = 'cuda:0'

class pose_dataset(data.Dataset):
    
    def __init__(self, mode='train'):
        
        self.mode = mode
        self.dataset = None
        
        filename = "database.pkl"
        if(self.mode == 'test'):
            filename = "database2.pkl"
        
        with open(filename, 'rb') as pfile:
            self.dataset = pickle.load(pfile)
                        
    def __getitem__(self, index):
        
        current_data_sample = self.dataset[index]
        
        image_wrist = torch.from_numpy(current_data_sample[0][0]).float()
        image_ls = torch.from_numpy(current_data_sample[0][1]).float()
        image_rs = torch.from_numpy(current_data_sample[0][2]).float()
        
        crackers_grasp_point_pose = current_data_sample[1][0]
        soup_grasp_point_pose = current_data_sample[1][1]
        tuna_grasp_point_pose = current_data_sample[1][2]
        
        crackers_xyz = torch.from_numpy(crackers_grasp_point_pose[:3]).float()
        soup_xyz = torch.from_numpy(soup_grasp_point_pose[:3]).float()
        tuna_xyz = torch.from_numpy(tuna_grasp_point_pose[:3]).float()
        
        targets = torch.cat((crackers_xyz, soup_xyz, tuna_xyz), 0)

        return image_wrist.permute(2,0,1), image_ls.permute(2,0,1), image_rs.permute(2,0,1), targets
    
    def __len__(self):
        return len(self.dataset)
        
        
class net(nn.Module):
    
    def __init__(self, target_length=9):
        super(net, self).__init__()
        
        in_layers = 512
        self.relu = nn.ReLU(inplace=True)
        self.model = nn.Sequential(*list(models.resnet18(pretrained=True).children()))[:-1]

        
        self.lin1 = nn.Linear(in_layers*3, 128)
        self.lin2 = nn.Linear(128, 9)
        
    def forward(self, x_wrist, x_ls, x_rs):
        
        x_wrist = self.model(x_wrist)
        x_ls = self.model(x_ls)
        x_rs = self.model(x_rs)
        
        x = torch.cat((x_wrist, x_ls, x_rs), 1).squeeze()
        x = self.lin2(self.relu(self.lin1(x)))
        
        return x
        
        
if __name__ == "__main__":
        
    model = net()
            
    train_dataset = pose_dataset()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)

    test_dataset = pose_dataset('test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    model = model.to(device)
    best_loss = 1000000

    for epoch in range(10):
        model.train()
        
        for i, (image_wrist, image_ls, image_rs, targets) in enumerate(train_loader):
            
            image_wrist, image_ls, image_rs, targets = image_wrist.to(device), image_ls.to(device), image_rs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            predictions = model(image_wrist, image_ls, image_rs)
            
            loss = criterion(predictions, targets)
            
            loss.backward()
            optimizer.step()
            
            if(i%10 == 0):
                print("Iter:", i, "/", len(train_loader), " | ", "Loss:", loss.item())

        model.eval()            
        total_loss = 0
        for i, (image_wrist, image_ls, image_rs, targets) in enumerate(test_loader):
            
            image_wrist, image_ls, image_rs, targets = image_wrist.to(device), image_ls.to(device), image_rs.to(device), targets.to(device)
            predictions = model(image_wrist, image_ls, image_rs)
            
            loss = criterion(predictions, targets)
            total_loss += loss.item()
            
            print("Validation .. Iter:", i, "/", len(test_loader), " | ", "Loss:", loss.item())
            
            # print(targets[0], predictions[0])
                
        if(total_loss < best_loss):
            best_loss = total_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Model Saved!")
        print("Total Loss:", total_loss)
            
            

        
            
        
        
        
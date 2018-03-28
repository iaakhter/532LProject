import pickle
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
from skimage import io, color, exposure
import torch

class spatial_dataset(Dataset):  
    def __init__(self, phase_dic, mag_dic, mode, transform=None):
 
        self.phase_keys = phase_dic.keys()
        self.phase_values = phase_dic.values()
        self.mag_keys = mag_dic.keys()
        self.mag_values = mag_dic.values()
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.phase_keys)

    def load_ucf_image(self, phase_video_name, mag_video_name):

        img = torch.FloatTensor(2, 512, 512)
        
        phase_img = Image.open(phase_video_name)
        phase_img = phase_img.convert('L')
        mag_img = Image.open(mag_video_name)
        mag_img = mag_img.convert('L')

        transformed_phase_img = self.transform(phase_img)
        transformed_mag_img = self.transform(mag_img)
        
        img[0, :, :] = transformed_phase_img
        img[1, :, :] = transformed_mag_img
        
        phase_img.close()
        mag_img.close()

        return img

    def __getitem__(self, idx):

        phase_video_name = self.phase_keys[idx]
        name_1 = phase_video_name.split('phase_512')[0]
        name_2 = phase_video_name.split('phase_512')[1]
        mag_video_name = name_1 + 'mag_512' + name_2

        label = self.phase_values[idx]
        
        if self.mode=='train':
            data = self.load_ucf_image(phase_video_name, mag_video_name)  
            sample = (data, label)
        elif self.mode=='val':
            data = self.load_ucf_image(phase_video_name, mag_video_name)
            sample = (phase_video_name, data, label)
        else:
            raise ValueError('There are only train and val mode')
           
        return sample

class spatial_dataloader():
    def __init__(self, BATCH_SIZE, num_workers, phase_train_list, phase_test_list, mag_train_list, mag_test_list):

        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers=num_workers
        self.phase_train_video, self.phase_test_video = {}, {}
        self.mag_train_video, self.mag_test_video = {}, {}
        self.phase_train_list = phase_train_list
        self.phase_test_list = phase_test_list
        self.mag_train_list = mag_train_list
        self.mag_test_list = mag_test_list


    def run(self):
        self.get_training_dic()
        self.get_testing_dic()
        train_loader = self.train()
        val_loader = self.validate()

        return train_loader, val_loader, self.phase_test_video

    def get_training_dic(self):
        #print '==> Generate frame numbers of each training video'
        lines_phase = [line.rstrip('\n') for line in open(self.phase_train_list)]
        for line in lines_phase:
            file_name = line.split(' ')[0]
            label = int(line.split(' ')[1])
            self.phase_train_video[file_name] = label
        
        lines_mag = [line.rstrip('\n') for line in open(self.mag_train_list)]
        for line in lines_mag:
            file_name = line.split(' ')[0]
            label = int(line.split(' ')[1])
            self.mag_train_video[file_name] = label
        
        #for i in range(10):
        #    phase_filename = lines_phase[i].split(' ')[0]
        #    mag_filename = lines_mag[i].split(' ')[0]
        #    print phase_filename
        #    print mag_filename
        #raise Exception("here")
            
           
                    
    def get_testing_dic(self):
        #print '==> sampling testing frames'
        lines = [line.rstrip('\n') for line in open(self.phase_test_list)]
        for line in lines:
            file_name = line.split(' ')[0]
            label = int(line.split(' ')[1])
            self.phase_test_video[file_name] = label
            
        lines = [line.rstrip('\n') for line in open(self.mag_test_list)]
        for line in lines:
            file_name = line.split(' ')[0]
            label = int(line.split(' ')[1])
            self.mag_test_video[file_name] = label

    def train(self):
        training_set = spatial_dataset(phase_dic=self.phase_train_video, mag_dic=self.mag_train_video, mode='train', transform = transforms.Compose([
                #transforms.Scale(224),
                #transforms.RandomCrop(224),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        print '==> Training data :',len(training_set),'frames'
        #print training_set[1][0]['img1'].size()

        train_loader = DataLoader(
            dataset=training_set, 
            batch_size=self.BATCH_SIZE,
            #shuffle=True,
            num_workers=self.num_workers)
        return train_loader

    def validate(self):
        validation_set = spatial_dataset(phase_dic=self.phase_test_video, mag_dic=self.mag_test_video, mode='val', transform = transforms.Compose([
                #transforms.Scale(224),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        
        print '==> Validation data :',len(validation_set),'frames'
        #print validation_set[1][1].size()

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers)
        return val_loader

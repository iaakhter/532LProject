import pickle
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
from skimage import io, color, exposure
#import cv2

class spatial_dataset(Dataset):  
    def __init__(self, dic, mode, transform=None):
 
        self.keys = dic.keys()
        self.values=dic.values()
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def load_ucf_image(self, video_name):

        img = Image.open(video_name)
        img = img.convert('L')
        img = img.crop((0, 0, 12, 128))
        #raise Exception("print shape")
        transformed_img = self.transform(img)
        img.close()

        return transformed_img

    def __getitem__(self, idx):

        if self.mode == 'train':
            video_name = self.keys[idx]
            
        elif self.mode == 'val':
            video_name = self.keys[idx]
        else:
            raise ValueError('There are only train and val mode')

        label = self.values[idx]
        
        if self.mode=='train':
            data = self.load_ucf_image(video_name)  
            #print data
            sample = (data, label)
        elif self.mode=='val':
            data = self.load_ucf_image(video_name)
            sample = (video_name, data, label)
        else:
            raise ValueError('There are only train and val mode')
           
        return sample

class spatial_dataloader():
    def __init__(self, BATCH_SIZE, num_workers, train_list, test_list):

        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers=num_workers
        self.train_video, self.test_video = {}, {}
        self.train_list = train_list
        self.test_list = test_list


    def run(self):
        self.get_training_dic()
        self.get_testing_dic()
        train_loader = self.train()
        val_loader = self.validate()

        return train_loader, val_loader, self.test_video

    def get_training_dic(self):
        #print '==> Generate frame numbers of each training video'
        lines = [line.rstrip('\n') for line in open(self.train_list)]
        for line in lines:
            file_name = line.split(' ')[0]
            label = int(line.split(' ')[1])
            self.train_video[file_name] = label
                    
    def get_testing_dic(self):
        #print '==> sampling testing frames'
        lines = [line.rstrip('\n') for line in open(self.test_list)]
        for line in lines:
            file_name = line.split(' ')[0]
            label = int(line.split(' ')[1])
            self.test_video[file_name] = label

    def train(self):
        training_set = spatial_dataset(dic=self.train_video, mode='train', transform = transforms.Compose([
                #transforms.Scale([224,224]),
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
            shuffle=True,
            num_workers=self.num_workers)
        return train_loader

    def validate(self):
        validation_set = spatial_dataset(dic=self.test_video, mode='val', transform = transforms.Compose([
                #transforms.Scale([224,224]),
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





#if __name__ == '__main__':
#    
#    dataloader = spatial_dataloader(BATCH_SIZE=1, num_workers=1, 
#                                path='/home/ubuntu/data/UCF101/spatial_no_sampled/', 
#                                #ucf_list='/home/ubuntu/cvlab/pytorch/ucf101_two_stream/github/UCF_list/',
#                                ucf_split='01')
#    train_loader,val_loader,test_video = dataloader.run()
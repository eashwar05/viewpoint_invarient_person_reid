import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from multiprocessing import freeze_support
import matplotlib.pyplot as plt

class VIPeRDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, train_ratio=0.7, extension='.bmp'):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        
        self.camA_dir = os.path.join(root_dir, 'cam_a')
        self.camB_dir = os.path.join(root_dir, 'cam_b')
        
        files = sorted([f for f in os.listdir(self.camA_dir) if f.endswith(extension)])
        
        samples_all = []
        for f in files:
            person_id = f.split('_')[0]
            fileA = os.path.join(self.camA_dir, f)
            fileB = os.path.join(self.camB_dir, f)
            if os.path.exists(fileA) and os.path.exists(fileB):
                samples_all.append((fileA, fileB, person_id))
        
        unique_ids = sorted(set([sample[2] for sample in samples_all]), key=lambda x: int(x))
        total_ids = len(unique_ids)
        split_index = int(total_ids * train_ratio)
        if self.split == 'train':
            selected_ids = unique_ids[:split_index]
        else:
            selected_ids = unique_ids[split_index:]
        
        self.id2label = {pid: idx for idx, pid in enumerate(selected_ids)}
        
        self.samples = []
        for sample in samples_all:
            if sample[2] in selected_ids:
                self.samples.append((sample[0], sample[1], self.id2label[sample[2]]))
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        fileA, fileB, label = self.samples[index]
        imgA = Image.open(fileA).convert('RGB')
        imgB = Image.open(fileB).convert('RGB')
        if self.transform:
            imgA = self.transform(imgA)
            imgB = self.transform(imgB)
        return imgA, imgB, label

def pad_to_target(img, target_size=224, fill=0):
    w, h = img.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img.resize((new_w, new_h), resample=Image.LANCZOS)
    
    new_img = Image.new("RGB", (target_size, target_size), (fill, fill, fill))
    left = (target_size - new_w) // 2
    top = (target_size - new_h) // 2
    new_img.paste(img_resized, (left, top))
    return new_img

def verify_preprocessing(loader):
    print("Verifying preprocessing...")
    for i, (imgA_batch, imgB_batch, labels) in enumerate(loader):
        print(f"Batch {i+1}:")
        print("camA batch shape:", imgA_batch.shape)
        print("camB batch shape:", imgB_batch.shape)
        print("Labels:", labels)

        # Visualize first 3 image pairs from the batch
        for j in range(min(3, len(imgA_batch))):
            fig, axes = plt.subplots(1, 2)
            
            # cam A image
            imgA_np = imgA_batch[j].permute(1, 2, 0).numpy()
            imgA_np_vis = (imgA_np * [0.229,0.224,0.225]) + [0.485,0.456,0.406]
            axes[0].imshow(imgA_np_vis.clip(0,1))
            axes[0].set_title(f"cam A - Label: {labels[j].item()}")
            axes[0].axis('off')

            # cam B image
            imgB_np = imgB_batch[j].permute(1, 2, 0).numpy()
            imgB_np_vis = (imgB_np * [0.229,0.224,0.225]) + [0.485,0.456,0.406]
            axes[1].imshow(imgB_np_vis.clip(0,1))
            axes[1].set_title(f"cam B - Label: {labels[j].item()}")
            axes[1].axis('off')

            plt.show()
        
        # Only verify one batch visually
        break

def main():
    train_transform = transforms.Compose([
        transforms.Resize((224, 84), interpolation=Image.LANCZOS),
        transforms.Lambda(lambda img: pad_to_target(img)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 84), interpolation=Image.LANCZOS),
        transforms.Lambda(lambda img: pad_to_target(img)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    root_dir = r"C:\Users\neash\Documents\Research Papers\VIPeR.v1.0\VIPeR"
    
    train_dataset = VIPeRDataset(root_dir=root_dir,
                                 split='train',
                                 transform=train_transform,
                                 train_ratio=0.7,
                                 extension='.bmp')
    
    test_dataset = VIPeRDataset(root_dir=root_dir,
                                split='test',
                                transform=test_transform,
                                train_ratio=0.7,
                                extension='.bmp')
    
    print("Number of training samples:", len(train_dataset))
    print("Number of testing samples:", len(test_dataset))
    
    train_loader = DataLoader(train_dataset,
                              batch_size=32,
                              shuffle=True,
                              num_workers=0)
    
    test_loader = DataLoader(test_dataset,
                             batch_size=32,
                             shuffle=False,
                             num_workers=0)
    
    # Check one batch to confirm correct batch size and shapes
    for imgA_batch,imgB_batch,label_batch in train_loader:
        print("camA batch shape:",imgA_batch.shape) # should be [32,...]
        print("camB batch shape:",imgB_batch.shape) # should be [32,...]
        print("Labels:",label_batch)
        break
    
    # Verify preprocessing visually
    verify_preprocessing(train_loader)

if __name__ == '__main__':
    freeze_support()
    main()





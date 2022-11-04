import cv2
import time
import torch
import torchvision
from torchsummary import summary
import torchvision.transforms as transforms
import numpy as np

from preprocess import CARLAPreprocess
from model import SteeringModel

torch.manual_seed(0)
np.random.seed(0)

SEED = 42
ROOT = "output/"
MIN, MAX = 0, 0
WEIGHT_DECAY = 1e-6
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 1
EPOCHS = 50
LR = 1e-4
CHECKPOINT_EPOCH = 0
BEST_LOSS = 1e10
LOAD_MODEL = False
SAVE_MODEL = True
TRAINING =  True

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    dataset = CARLAPreprocess(transform=transform)
    # MIN, MAX = dataset.get_min_max()  
    train, val = torch.utils.data.random_split(dataset, [int(len(dataset)*(1-VALIDATION_SPLIT)), int(len(dataset)*VALIDATION_SPLIT)+1])
    trainloader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True) 
    valloader = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE, shuffle=False)

    # steering_model = SteeringModel().to(device)
    # optimizer = torch.optim.Adam(steering_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # criterion = torch.nn.MSELoss()
    # summary(steering_model, (3, 128, 256))

    # if LOAD_MODEL:
    #     print("Loading models...")
    #     checkpoint = torch.load("saved_models/steering.pth")
    #     steering_model.load_state_dict(checkpoint['model_state'])
    #     optimizer.load_state_dict(checkpoint['optim_state'])
    #     model_loss = checkpoint['loss']
    #     CHECKPOINT_EPOCH = checkpoint['epoch']+1
    #     BEST_LOSS = model_loss
    #     print("Done!")
    #     print('-'*20) 

    # print(len(dataset))
    # img, values = next(iter(trainloader))
    # print(img, values)

    from time import time
    import multiprocessing as mp
    for num_workers in range(2, mp.cpu_count(), 2):  
        train_loader = torch.utils.data.DataLoader(dataset,shuffle=True,num_workers=num_workers,batch_size=64,pin_memory=True)
        start = time()
        for epoch in range(1, 3):
            for i, data in enumerate(train_loader, 0):
                pass
        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))


    # print("Starting training...")
    # for epoch in range(CHECKPOINT_EPOCH, CHECKPOINT_EPOCH+EPOCHS):

    #     steering_model.train()
    #     for idx, (img,angle) in enumerate(trainloader):
    #         optimizer.zero_grad()
    #         img = img.to(device, dtype=torch.float)
    #         angle = angle.to(device, dtype=torch.float).view(-1)
    #         output = steering_model(img).view(-1)
    #         loss = criterion(output, angle)
    #         loss.backward()
    #         optimizer.step()

    #         if idx % (len(trainloader)//5) == 0:
    #             print(f"Epoch: [{epoch+1}/{CHECKPOINT_EPOCH+EPOCHS}] Index: [{idx}/{len(trainloader)}] Loss: {loss.item()}")
        
    #     running_val_loss = 0.0
    #     steering_model.eval() 
    #     with torch.no_grad(): 
    #         for idx, (img,angle) in enumerate(valloader, 0):
    #             img = img.to(device, dtype=torch.float)
    #             angle = angle.to(device, dtype=torch.float).view(-1)
    #             output = steering_model(img).view(-1)
    #             loss = criterion(output, angle)
    #             running_val_loss += loss.item()
    #     print(f"Validation Loss: {running_val_loss/len(valloader)}")

    #     if (running_val_loss/len(valloader)) < BEST_LOSS: 
    #         if SAVE_MODEL:
    #             print("Saving model...")
    #             torch.save({
    #                 'epoch': epoch,
    #                 'model_state': steering_model.state_dict(),
    #                 'optim_state': optimizer.state_dict(),
    #                 'loss': loss.item(),
    #             }, f"saved_models/steering_64bs.pth")
    #             print("Done!")
    #             BEST_LOSS = running_val_loss
    #             print('-'*20)

    # else:
    #     dataset = SteeringDataset(ROOT, CROP)
    #     MIN, MAX = dataset.get_min_max()
    #     i = 0
    #     smoothed_angle = 0
    #     wheel = cv2.imread("steering_wheel.png",0)
    #     h, w = wheel.shape
    #     steering_model = SteeringModel().to(device)
    #     print("Loading models...")
    #     checkpoint = torch.load("saved_models/steering_64bs.pth")
    #     steering_model.load_state_dict(checkpoint['model_state'])
    #     while (cv2.waitKey(10) != ord('q')) or i<=100:
    #         # img = cv2.imread("steering/data/"+str(i)+".jpg")
    #         img = cv2.imread(ROOT+str(i)+".jpg")
    #         process = cv2.resize(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),(200,66))[CROP:,:] #reading in RGB
    #         process = process/255.0
    #         angle = steering_model(Img2Tensor(process,device))
    #         angle = (angle.item()*0.5+0.5)*(MAX-MIN)+MIN
    #         smoothed_angle += 0.2 * pow(abs((angle - smoothed_angle)), 2.0/3.0) * (angle - smoothed_angle) / abs(angle - smoothed_angle)
    #         dst = cv2.warpAffine(wheel,cv2.getRotationMatrix2D((w/2,h/2),-smoothed_angle,1),(w,h))
    #         dst = cv2.putText(dst, f"Predicted angle: {angle:.2f} degrees.", (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    #         canny = cv2.Canny(image=img, threshold1=100, threshold2=200)
    #         cv2.imshow("frame", img)
    #         cv2.imshow("processed_image", process)
    #         cv2.imshow("canny", canny)
    #         cv2.imshow("steering_wheel", dst)
    #         # time.sleep(0.25)
    #         i += 1
    #     cv2.destroyAllWindows()
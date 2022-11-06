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
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-2
CHECKPOINT_EPOCH = 0
BEST_LOSS = 1e10
LOAD_MODEL = False
SAVE_MODEL = True
TRAINING =  True

"""
BASED OFF OUR DATA, WE ASSUME THAT THE BRAKE AND 
"""

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    transformer = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    dataset = CARLAPreprocess(transform=transformer)
    # MIN, MAX = dataset.get_min_max()  
    train, val = torch.utils.data.random_split(dataset, [int(len(dataset)*(1-VALIDATION_SPLIT)), int(len(dataset)*VALIDATION_SPLIT)+1])
    trainloader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True) #tested and confirmed that num_worked=2 works best for me: https://chtalhaanwar.medium.com/pytorch-num-workers-a-tip-for-speedy-training-ed127d825db7
    valloader = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE, shuffle=False)

    steering_model = SteeringModel().to(device)
    optimizer = torch.optim.Adam(steering_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion1 = torch.nn.MSELoss()
    criterion2 = torch.nn.BCELoss()

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


    # print("Starting training...")
    for epoch in range(CHECKPOINT_EPOCH, CHECKPOINT_EPOCH+EPOCHS):

        steering_model.train()
        for idx, (img,truth) in enumerate(trainloader):
            optimizer.zero_grad()
            img = img.to(device, dtype=torch.float)
            truth = truth.to(device, dtype=torch.float)

            pred = steering_model(img).to(device)

            # print(pred[0], truth[0].clone())
            # print(pred[:,1:].clone(), truth[:,1:].clone())

            loss = criterion1(pred, truth)

            # loss1 = criterion1(pred[:,0], truth[:,0]) #steering difference
            # loss2 = criterion2(pred[:,1:], pred[:,1:]) #binary classification on all others
            
            # loss = loss1 + loss2 #take sum of losses and backpropogate w.r.t. that
            # loss1.backward()
            # loss2.backward()
            loss.backward()
            optimizer.step()

            if idx % (len(trainloader)//5) == 0:
                print(f"Epoch: [{epoch+1}/{CHECKPOINT_EPOCH+EPOCHS}] Index: [{idx}/{len(trainloader)}] Cumulative Loss: {loss.item()}")#Steering Loss: {loss1.item()} Throttle, Brake, Reverse Loss: {loss2.item()}")


        # steering_model.train()
        # for idx, (img,truth) in enumerate(trainloader):
        #     optimizer.zero_grad()
        #     img = img.to(device, dtype=torch.float)
        #     truth = truth.to(device, dtype=torch.float).view(-1)

        #     steer, other = steering_model(img)
        #     steer = steer.to(device)
        #     other = other.to(device)

        #     # print(pred[0], truth[0].clone())
        #     # print(pred[1:].clone(), truth[1:].clone())

        #     loss1 = criterion1(steer[0], truth[0]) #steering difference
        #     loss2 = criterion2(other[1:], truth[1:]) #binary classification on all others
            
        #     loss = loss1 + loss2 #take sum of losses and backpropogate w.r.t. that
        #     # loss1.backward()
        #     # loss2.backward()
        #     loss.backward()
        #     optimizer.step()
        
        running_val_loss = 0.0
        steering_model.eval() 
        with torch.no_grad(): 
            for idx, (img,truth) in enumerate(valloader, 0):
                img = img.to(device, dtype=torch.float)
                truth = truth.to(device, dtype=torch.float)
                pred = steering_model(img)
                loss = criterion1(pred, truth)
                # loss2 = criterion2(other[1:], truth[1:])
                running_val_loss += loss.item()
        print(f"Validation Loss: {running_val_loss/len(valloader)}")

        print(running_val_loss/len(valloader), BEST_LOSS)

        if (running_val_loss/len(valloader)) < BEST_LOSS: 
            if SAVE_MODEL:
                print("Saving model...")
                torch.save({
                    'epoch': epoch,
                    'model_state': steering_model.state_dict(),
                    'optim_state': optimizer.state_dict(),
                    'loss': loss.item(),
                    # 'loss2': loss.item(),
                }, "steering.pth")
                print("Done!")
                BEST_LOSS = running_val_loss
                print('-'*20)

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
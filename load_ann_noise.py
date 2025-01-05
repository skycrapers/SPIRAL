import torch
import logging
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import random
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from time import *
from torch.autograd import Variable

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#torch.manual_seed(123)

class net(nn.Module):
    def __init__(self, inlen):
        super(net,self).__init__()
        self.fc0 = nn.Linear(inlen*2, inlen)
        self.fc1 = nn.Linear(inlen+3, 18)
        self.fc2 = nn.Linear(18, 18)
        self.fc3 = nn.Linear(18, 18)
        self.fc4 = nn.Linear(18, 1)
        
    def forward(self,x,xx,p1,p2,p3):
        out1 = torch.tanh(self.fc0(torch.cat([x,xx],dim=1)))
        out2 = torch.tanh(self.fc1(torch.cat([out1,p1,p2/50.0,p3],dim=1)))
        out3 = torch.tanh(self.fc2(out2))
        out4 = torch.tanh(self.fc3(out3))
        out5 = (self.fc4(out4))
        out = out5.unsqueeze(dim=1)

        return out

###################################################################################################

#train
def train(xnum, type, mid, datadir, seqlen, samrate, inlen):
    is_load=True
    train_batch_size = 1
    filename = datadir+'/train.txt'
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    numberOfOutputDimension = 1 
    if is_load==True:
        data = np.zeros((numberOfLines, 2, seqlen+3)) 
    else:
        data = np.zeros((numberOfLines, 2, seqlen)) 
    labela = np.zeros((numberOfLines, 1, seqlen))

    for index in range(len(arrayOLines)):
        line = arrayOLines[index]
        listFromLine = np.load(datadir+'/'+re.split('\n', line)[0]+'.npy')
        data[index,0,:] = listFromLine[0,:]
        data[index,1,:] = listFromLine[1,:]
        labela[index,0,:] = 10*listFromLine[2,0:seqlen]
    numberOfTrainingData = numberOfLines
    xTrain = data[:numberOfTrainingData]
    yTrain = labela[:numberOfTrainingData]
    
    if seqlen%samrate == 0:
        sam_seqlen=seqlen//samrate
    else:
        sam_seqlen=seqlen//samrate+1
    if is_load==True:
        xtrainsam = np.zeros([numberOfTrainingData,2,sam_seqlen+3]) 
        xtrainsam[:,0,sam_seqlen:sam_seqlen+3]=xTrain[:,0,seqlen:seqlen+3]
    else:
        xtrainsam = np.zeros([numberOfTrainingData,2,sam_seqlen]) 
    
    ytrainsam = np.zeros([numberOfTrainingData,1,sam_seqlen])
    for i in range(sam_seqlen):
        xtrainsam[:,:,i]=xTrain[:,:,i*samrate]
        ytrainsam[:,:,i]=yTrain[:,:,i*samrate]

    xTrain=xtrainsam
    yTrain=ytrainsam
    xTrain=torch.Tensor(xTrain)
    yTrain=torch.Tensor(yTrain)
    train_dataset=TensorDataset(xTrain,yTrain)
    trainloader=DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=0)
    
    label.begin
    start_epoch = 0
    train_epoch=800    
    model=net(inlen).cuda()
    if start_epoch != 0:
        model.load_state_dict(torch.load('./checkpoint/'+mid+'/'+mid+'.pth'))
    criterion=nn.MSELoss(reduction='sum')
    optimizer=torch.optim.SGD(model.parameters(), lr=0.0001, weight_decay=0) 
    
    isExists = os.path.exists('./checkpoint/'+mid+'/')
    if not isExists:
        os.makedirs('./checkpoint/'+mid+'/')
    loss_list=[]
    begin1=time()
    for epoch in range(start_epoch, train_epoch+start_epoch):
        running_loss=0.0
        for i,data in enumerate(trainloader):
            loss = 0.0
            inputs,labels=data[0].cuda(),data[1].cuda()
            model.train()
            optimizer.zero_grad()
            N, _, _ = inputs.size()
            
            if type=='all':
                outputs = torch.zeros(N,1,inlen-1+xnum).cuda()
                for t in range(inlen-1, inlen-1+xnum):
                    inp = inputs[:,0,t-(inlen-1):t+1]
                    outputs[:,:,t:t+1]=model(inp, inputs[:,1,t-(inlen-1):t+1], inputs[:,0,sam_seqlen:sam_seqlen+1], inputs[:,0,sam_seqlen+1:sam_seqlen+2], inputs[:,0,sam_seqlen+2:sam_seqlen+3])
                    
                loss = criterion(outputs[:,:,inlen-1:inlen-1+xnum],labels[:,:,inlen-1:inlen-1+xnum])
                loss.backward()
                optimizer.step() 
                running_loss+=loss.item()
                loss_list.append(loss.item())
                
                print('[%d, %5d] loss: %.5f' %(epoch+1, i+1, running_loss))
                running_loss=0.0 
                
        if epoch>0 and epoch%50==0:
            torch.save(model.state_dict(),'./checkpoint/'+mid+'/'+mid+'_'+str(epoch)+'ep.pth')
        
    end1=time()
    print('Finished Training')
    print(end1-begin1)
    torch.save(model.state_dict(),'./checkpoint/'+mid+'/'+mid+'.pth')

def inference(mid, tseqlen, samrate, inlen, testdata):   
    is_load = True
    testlabel = np.zeros((1, tseqlen))
    
    xTest = testdata
    yTest = testlabel

    if tseqlen%samrate == 0:
        sam_tseqlen=tseqlen//samrate
    else:
        sam_tseqlen=tseqlen//samrate+1
    
    if is_load==True: 
        xtestsam = np.zeros([1,2,sam_tseqlen+3])
        xtestsam[0,:,sam_tseqlen:sam_tseqlen+3]=xTest[:,tseqlen:tseqlen+3]
    else:
        xtestsam = np.zeros([1,2,sam_tseqlen])
    
    ytestsam = np.zeros([1,1,sam_tseqlen])

    for i in range(sam_tseqlen):
        xtestsam[0,:,i]=xTest[:,i*samrate]
        ytestsam[0,:,i]=yTest[:,i*samrate]

    xTest=xtestsam
    yTest=ytestsam
    xTest=torch.Tensor(xTest)
    yTest=torch.Tensor(yTest)
    test_dataset=TensorDataset(xTest,yTest)
    testloader=DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    isExists = os.path.exists('./output/')
    if not isExists:
        os.makedirs('./output/')
    modelt=net(inlen).cuda()
    modelt.load_state_dict(torch.load('./checkpoint/'+mid+'/'+mid+'.pth'))
    with open('./output/'+mid+'.txt','w') as f:       
        for data in testloader:
            inputs,labels=data[0].cuda(),data[1].cuda()
            N, C, LI = inputs.size()
            pinputs = (torch.zeros(N,C,LI+inlen-1)).cuda()
            pinputs[:,:,inlen-1:LI+inlen-1] = inputs[:,:,0:LI]
            pinputs[:,:,0:inlen-1] = inputs[:,:,0:1]
            
            paral_rate = 50
            outputs = torch.zeros(N,1,sam_tseqlen).cuda() ##
            with torch.no_grad():
                modelt.eval()               
                temp = 0
                for t in range(inlen-1, sam_tseqlen+inlen-1, paral_rate):
                    if t+paral_rate<=sam_tseqlen+inlen-1:
                        inp = torch.zeros(paral_rate,2,inlen).cuda()
                        for b in range(paral_rate):                        
                            inp[b,:,:] = pinputs[0,:,t-(inlen-1)+b:t+1+b]
                        outputs[:,:,t-(inlen-1):t-(inlen-2)+paral_rate-1]=modelt(inp[:,0,:], inp[:,1,:], inputs[:,0,sam_tseqlen:sam_tseqlen+1].repeat(paral_rate,1), inputs[:,0,sam_tseqlen+1:sam_tseqlen+2].repeat(paral_rate,1), inputs[:,0,sam_tseqlen+2:sam_tseqlen+3].repeat(paral_rate,1)).reshape(1,1,paral_rate)
                    else:
                        temp = t
                        break
                if temp != 0:
                    for t in range(temp, sam_tseqlen+inlen-1):
                        inp = pinputs[:,:,t-(inlen-1):t+1]
                        outputs[:,:,t-(inlen-1):t-(inlen-2)]=modelt(inp[:,0,:], inp[:,1,:], inputs[:,0,sam_tseqlen:sam_tseqlen+1], inputs[:,0,sam_tseqlen+1:sam_tseqlen+2], inputs[:,0,sam_tseqlen+2:sam_tseqlen+3])
            
            for n in range(N):
                output = outputs[n,0,0:sam_tseqlen].cpu().numpy()
                f.write(str(output)+'\n')
                
    if samrate != 1:
        out_i = np.zeros(tseqlen,)
        for i in range(output.shape[0]-1):
            for j in range(samrate):
                out_i[i*samrate+j] = output[i]+j*(output[i+1]-output[i])/samrate
        for k in range(tseqlen-(output.shape[0]-1)*samrate):
            out_i[(output.shape[0]-1)*samrate+k]=output[output.shape[0]-1]+(output[output.shape[0]-1]-output[output.shape[0]-2])/samrate*k
    else:
        out_i = output
    
    out_i = out_i/10
    
    return out_i  

#test
def test(type, mid, datadir, tseqlen, samrate, per, inlen, ep): 
    is_load=True
    testfile = './dataset/'+datadir+'/test508.txt'
    frt = open(testfile)
    tarrayOLines = frt.readlines()
    tnumberOfLines = len(tarrayOLines)
    if is_load==True:
        testdata = np.zeros((tnumberOfLines, 2, tseqlen+3)) 
    else:
        testdata = np.zeros((tnumberOfLines, 2, tseqlen))
    testlabel = np.zeros((tnumberOfLines, 1, tseqlen))

    for index in range(len(tarrayOLines)):
        tline = tarrayOLines[index]
        tlistFromLine = np.load('./dataset/'+datadir+'/'+re.split('\n', tline)[0]+'.npy')
        testdata[index,0,:] = tlistFromLine[0,:]
        testdata[index,1,:] = tlistFromLine[1,:]
        testlabel[index,0,:] = 10*tlistFromLine[2,0:tseqlen]
    
    numberOfTestData = tnumberOfLines
    xTest = testdata[:numberOfTestData]
    yTest = testlabel[:numberOfTestData]

    if tseqlen%samrate == 0:
        sam_tseqlen=tseqlen//samrate
    else:
        sam_tseqlen=tseqlen//samrate+1
    if is_load==True: 
        xtestsam = np.zeros([numberOfTestData,2,sam_tseqlen+3])
        xtestsam[:,0,sam_tseqlen:sam_tseqlen+3]=xTest[:,0,tseqlen:tseqlen+3]
    else:
        xtestsam = np.zeros([numberOfTestData,2,sam_tseqlen])
    
    ytestsam = np.zeros([numberOfTestData,1,sam_tseqlen])

    for i in range(sam_tseqlen):
        xtestsam[:,:,i]=xTest[:,:,i*samrate]
        ytestsam[:,:,i]=yTest[:,:,i*samrate]

    xTest=xtestsam
    yTest=ytestsam
    xTest=torch.Tensor(xTest)
    yTest=torch.Tensor(yTest)
    test_dataset=TensorDataset(xTest,yTest)
    testloader=DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    isExists = os.path.exists('./result/')
    if not isExists:
        os.makedirs('./result/')
    be2=time()
    err=0.0
    err_lists = []
    rerr_lists = []
    maxr_lists = []
    maxre_lists = []
    modelt=net(inlen).cuda()
    modelt.load_state_dict(torch.load('./checkpoint/'+mid+'/'+ep+'.pth'))
    with open('./result/'+mid+'.txt','w') as f:
        f.write(modelt.__str__())
        f.write('\n')
        flag = 0
        for data in testloader:
            flag = flag + 1
            inputs,labels=data[0].cuda(),data[1].cuda()
            N, C, LI = inputs.size()
            pinputs = torch.zeros(N,2,LI+inlen-1).cuda()
            pinputs[:,:,inlen-1:LI+inlen-1] = inputs[:,:,0:LI]
            pinputs[:,:,0:inlen-1] = inputs[:,:,0:1]
            if type=='period':
                endi=per
            elif type=='all':
                endi=sam_tseqlen
            
            inph, _ = torch.max(inputs[:,1:2,0:-3], dim=2)
            outputs = torch.zeros(N,1,inlen-1+endi).cuda()

            with torch.no_grad():
                modelt.eval()                               
                for t in range(inlen-1, endi+inlen-1): #
                    inp = pinputs[:,0,t-(inlen-1):t+1]
                    outputs[:,:,t:t+1]=modelt(inp, pinputs[:,1,t-(inlen-1):t+1], inputs[:,0,sam_tseqlen:sam_tseqlen+1], inputs[:,0,sam_tseqlen+1:sam_tseqlen+2], inputs[:,0,sam_tseqlen+2:sam_tseqlen+3], )
                                        
        
            for n in range(N):
                label = (labels[n,0,inlen:endi]/10.0).cpu().numpy() #
                output = (outputs[n,0,inlen+inlen-1:inlen-1+endi]/10.0).cpu().numpy() #
                input_ = inputs[n,0,inlen:endi].cpu().numpy()
                cload = float(inputs[n,0,sam_tseqlen].cpu())
                rload = float(inputs[n,0,sam_tseqlen+1].cpu())
                voc_load = float(inputs[n,0,sam_tseqlen+2].cpu())

                # calculate err
                err_list = np.mean(abs(label - output))
                rerr_list = np.mean(abs(label - output)/inph[0,0].cpu().numpy())
                err_lists.append(err_list)
                rerr_lists.append(rerr_list)
            
                rnp = abs(label-output)
                maxerr = np.max(rnp)
                maxre = np.max(abs(label - output)/inph[0,0].cpu().numpy())
                maxr_lists.append(maxerr)
                maxre_lists.append(maxre)
                
                print('err: {:.4f} | rerr: {:.4f} | maxerr: {:.4f} | maxre: {:.4f}'.format(err_list, rerr_list, maxerr, maxre))
                f.write('err: {:.4f} | rerr: {:.4f} | maxerr: {:.4f} | maxre: {:.4f}\n'.format(err_list, rerr_list, maxerr, maxre))
                
        print('{:10} err: {:.4f} | rerr: {:.4f} | maxerr: {:.4f} | maxre: {:.4f}'.format(
            'Average', np.mean(err_lists), np.mean(rerr_lists), np.mean(maxr_lists), np.mean(maxre_lists)))
        f.write('{:10} err: {:.4f} | rerr: {:.4f} | maxerr: {:.4f} | maxre: {:.4f}\n'.format(
            'Average', np.mean(err_lists), np.mean(rerr_lists), np.mean(maxr_lists), np.mean(maxre_lists)))
        
    out_i = np.zeros(tseqlen,)
    in_i = np.zeros(tseqlen,)
    for i in range(output.shape[0]-1):
        for j in range(samrate):
            out_i[i*samrate+j] = output[i]+j*(output[i+1]-output[i])/samrate
            in_i[i*samrate+j] = input_[i]+j*(input_[i+1]-input_[i])/samrate
    for k in range(tseqlen-(output.shape[0]-1)*samrate):
        out_i[(output.shape[0]-1)*samrate+k]=output[output.shape[0]-1]+(output[output.shape[0]-1]-output[output.shape[0]-2])/samrate*k   
        in_i[(output.shape[0]-1)*samrate+k]=input_[output.shape[0]-1]+(input_[output.shape[0]-1]-input_[output.shape[0]-2])/samrate*k           
    
    end2=time()
    print(end2-be2)
    
    return in_i, out_i, cload, rload, voc_load

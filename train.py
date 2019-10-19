# coding=utf-8

from sky.hrnet import cls_net
from sky.preprocess import get_data
import sky.config 

import torch.optim as optim
import time

def train():
    # get data
    trainloader = get_data(config.data_path)
    # get HRNet
    hrnet = cls_net(config.net)
    # define a optimizer
    optimizer = optim.Adam(net.parameters(), lr=config.lr)
    # load model if exists
    if os.path.exists(config.model_load_path) is not True:
        criterion = nn.CrossEntropyLoss()
    else:
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        initepoch = checkpoint['epoch']
        criterion = checkpoint['loss']

    # train
    for epoch in range(config.epoch): 
        timestart = time.time()

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            # forward, backward and step
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
                # get accuaracy of train dataset
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                print('Accuracy of the network on the %d tran images: %.3f %%' % (total,
                        100.0 * correct / total))
                total = 0
                correct = 0
                # save the model
                torch.save({'epoch':epoch,
                            'model_state_dict':net.state_dict(),
                            'optimizer_state_dict':optimizer.state_dict(),
                            'loss':criterion
                            }, config.model_save_path)

        print('epoch %d cost %3f sec' %(epoch,time.time()-timestart))

    print('Finished Training')

if __name__ == "__main__":
    train()
from cvt import *
import torch
from torch import nn, einsum
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import v2


def train_loop(loss_fn, optim, trainloader, net):
    running_loss = 0
    for i, data in enumerate(trainloader, 0):
        
        inputs, labels = data
        inputs = inputs
        labels = labels
        # zero parameter gradients
        optim.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optim.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print(f'loss: {running_loss / 100:.3f}')
            running_loss = 0.0
        print(loss)

def train(epochs, trainloader, net, loss_fn=nn.CrossEntropyLoss(), optimizer=optim.Adam(cvt.parameters(),lr=0.001)):
    net.train()
    for i in range(epochs):
        train_loop(loss_fn, optimizer, trainloader, net)
    print("Done")

def test(model, loss_fn, test_loader):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    for i, data in enumerate(test_loader, 1):
        inputs, labels = data
        inputs = inputs
        labels = labels
        pred = model(inputs)
        loss = loss_fn(pred, labels)

        running_loss += loss.item()
        _, predicted = pred.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / len(test_loader)
    accuracy = 100. * correct / total

    print('Test Loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        avg_loss, correct, total, accuracy))


transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Resize((224, 224))])

food101_train_data = datasets.Food101('/', split="train", download=True, transform=transform)
food101_test_data = datasets.Food101('/', split="test", transform=transform)

train_loader = torch.utils.data.DataLoader(food101_train_data,
                                          batch_size=32,
                                          shuffle=True)

cvt = CvT(224, 3, 101)

def train_loop(loss_fn, optim, trainloader, net):
    running_loss = 0
    for i, data in enumerate(trainloader, 0):
        
        inputs, labels = data
        inputs = inputs
        labels = labels
        # zero parameter gradients
        optim.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optim.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print(f'loss: {running_loss / 100:.3f}')
            running_loss = 0.0
        print(loss)



optimizer = torch.optim.Adam(cvt.parameters(), lr=1e-3)
train(1, train_loader, cvt, optimizer=optimizer)

# Save model
PATH = "cvt_impl.pt"
torch.save(cvt.state_dict(), PATH)

test_loader = torch.utils.data.DataLoader(food101_test_data,
                                          batch_size=64,
                                          shuffle=True)

test(cvt, nn.CrossEntropyLoss(), test_loader)
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.models import inception_v3, resnet18
from torch.utils.data import DataLoader
"""
Trying to do a forward pass of MNIST images through the resnet18 network.
Next step: modify remove output layer to be left with a final embedding.
Problem: data inputs are currently missing 3-channel dimension. 
"""

def main():
	# use any torchvision network here
	net = resnet18(pretrained=True)
	print('loaded net.')

	# load MNIST dataset
	transform = transforms.Compose(
    	[transforms.ToTensor(),
    	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	train_dataset = datasets.MNIST("mnist", train=True, download=True, transform=transform)
	test_dataset = datasets.MNIST("mnist", train=False, download=True, transform=transform)
	train_dl = DataLoader(train_dataset, batch_size=4, shuffle=True)
	test_dl = DataLoader(test_dataset, batch_size=4, shuffle=True)

	assert len(train_dataset) == 60000
	assert len(test_dataset) == 10000

	iter_dataset = iter(train_dl)
	inputs, labels = next(iter_dataset) # inputs: torch.Size([4, 1, 28, 28])

	# pass the first batch through the network
	if torch.cuda.is_available():
		net = net.cuda()
		inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())   
	else:
		inputs, labels = Variable(inputs), Variable(labels)

	print(net(inputs))

main()

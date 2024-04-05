class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.flatten = nn.Flatten()

        in_channel = 1
        out_channel = 16
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding ='same') # bias = False
        self.act1 = nn.ReLU(inplace= True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Extra convolutional layer:
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding ='same') # bias = False
        self.act2 = nn.ReLU(inplace= True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # We had to recalculate the number of neurons after flattening (784):
        self.linear2 = nn.Linear(784,392)
        self.act2 = nn.ReLU(inplace= True)
        self.linear3 = nn.Linear(392, 10)

    def forward(self, x):

        self.feature_maps = self.conv1(x)
        tmp = self.act1(self.feature_maps)
        tmp = self.maxpool1(tmp)

        # Extra convolutional layer:
        tmp = self.conv2(tmp)
        tmp = self.act2(tmp)
        tmp = self.maxpool2(tmp)

        tmp = torch.flatten(tmp, 1, -1)
        tmp = self.linear2(tmp)
        tmp = self.act2(tmp)
        logits = self.linear3(tmp)

        return logits

model = CNN()
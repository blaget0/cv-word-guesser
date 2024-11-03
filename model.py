import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from tqdm import tqdm
import numpy as np
import crop
import time
import files
from collections import OrderedDict

classes = ['aircraft carrier', 'airplane', 'alarm clock', 'ambulance', 'angel', 'animal migration', 'ant', 'anvil', 'apple', 'arm', 'asparagus', 'axe', 'backpack', 
           'banana', 'bandage', 'barn', 'baseball bat', 'baseball', 'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed', 'bee', 'belt', 'bench', 
           'bicycle', 'binoculars', 'bird', 'birthday cake', 'blackberry', 'blueberry', 'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 
           'broccoli', 'broom', 'bucket', 'bulldozer', 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar', 'camel', 'camera', 'camouflage', 'campfire', 
           'candle', 'cannon', 'canoe', 'car', 'carrot', 'castle', 'cat', 'ceiling fan', 'cell phone', 'cello', 'chair', 'chandelier', 'church', 'circle', 'clarinet', 'clock', 
           'cloud', 'coffee cup', 'compass', 'computer', 'cookie', 'cooler', 'couch', 'cow', 'crab', 'crayon', 'crocodile', 'crown', 'cruise ship', 'cup', 'diamond', 'dishwasher', 
           'diving board', 'dog', 'dolphin', 'donut', 'door', 'dragon', 'dresser', 'drill', 'drums', 'duck', 'dumbbell', 'ear', 'elbow', 'elephant', 'envelope', 'eraser', 
           'eye', 'eyeglasses', 'face', 'fan', 'feather', 'fence', 'finger', 'fire hydrant', 'fireplace', 'firetruck', 'fish', 'flamingo', 'flashlight', 'flip flops', 'floor lamp', 
           'flower', 'flying saucer', 'foot', 'fork', 'frog', 'frying pan', 'garden hose', 'garden', 'giraffe', 'goatee', 'golf club', 'grapes', 'grass', 'guitar', 'hamburger', 
           'hammer', 'hand', 'harp', 'hat', 'headphones', 'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey puck', 'hockey stick', 'horse', 'hospital', 'hot air balloon', 
           'hot dog', 'hot tub', 'hourglass', 'house plant', 'house', 'hurricane', 'ice cream', 'jacket', 'jail', 'kangaroo', 'key', 'keyboard', 'knee', 'knife', 'ladder', 
           'lantern', 'laptop', 'leaf', 'leg', 'light bulb', 'lighter', 'lighthouse', 'lightning', 'line', 'lion', 'lipstick', 'lobster', 'lollipop', 'mailbox', 'map', 'marker', 
           'matches', 'megaphone', 'mermaid', 'microphone', 'microwave', 'monkey', 'moon', 'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 
           'nail', 'necklace', 'nose', 'ocean', 'octagon', 'octopus', 'onion', 'oven', 'owl', 'paint can', 'paintbrush', 'palm tree', 'panda', 'pants', 'paper clip', 'parachute', 
           'parrot', 'passport', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'piano', 'pickup truck', 'picture frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers', 
           'police car', 'pond', 'pool', 'popsicle', 'postcard', 'potato', 'power outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'rain', 'rainbow', 'rake', 'remote control', 
           'rhinoceros', 'rifle', 'river', 'roller coaster', 'rollerskates', 'sailboat', 'sandwich', 'saw', 'saxophone', 'school bus', 'scissors', 'scorpion', 'screwdriver', 
           'sea turtle', 'see saw', 'shark', 'sheep', 'shoe', 'shorts', 'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping bag', 'smiley face', 'snail', 'snake', 
           'snorkel', 'snowflake', 'snowman', 'soccer ball', 'sock', 'speedboat', 'spider', 'spoon', 'spreadsheet', 'square', 'squiggle', 'squirrel', 'stairs', 'star', 'steak', 
           'stereo', 'stethoscope', 'stitches', 'stop sign', 'stove', 'strawberry', 'streetlight', 'string bean', 'submarine', 'suitcase', 'sun', 'swan', 'sweater', 'swing set', 
           'sword', 'syringe', 't-shirt', 'table', 'teapot', 'teddy-bear', 'telephone', 'television', 'tennis racquet', 'tent', 'The Eiffel Tower', 'The Great Wall of China', 
           'The Mona Lisa', 'tiger', 'toaster', 'toe', 'toilet', 'tooth', 'toothbrush', 'toothpaste', 'tornado', 'tractor', 'traffic light', 'train', 'tree', 'triangle', 'trombone', 
           'truck', 'trumpet', 'umbrella', 'underwear', 'van', 'vase', 'violin', 'washing machine', 'watermelon', 'waterslide', 'whale', 'wheel', 'windmill', 'wine bottle', 
           'wine glass', 'wristwatch', 'yoga', 'zebra', 'zigzag']

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        
        self.linear_block = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128*7*7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, len(classes))
        )
        
    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.linear_block(x)
        
        return x

t0 = time.time()

def predict(data, model_path, word):

    model = torch.load(model_path, weights_only=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    image = crop.crop_image(data)
    image = torch.FloatTensor(image).to(device)
    image = image.unsqueeze(0).unsqueeze(0).repeat(2,1,1,1)

    model.eval()
    with torch.no_grad():
        pred = model.forward(image)
        softmax = torch.nn.Softmax(dim=0)
        prob = softmax(pred[0])
        index = classes.index(word)

    return prob[index].detach().item()


if __name__ == '__main__':
    hidden_sizes = [128, 100, 64]
    output_size = len(classes)

    input_size = 28*28
    dropout = 0.0
    weight_decay = 0.0
    n_chunks = 700
    learning_rate = 0.03

    
    model2 = torch.nn.Sequential(OrderedDict([
                            ('conv1', torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, padding=1)),
                            ('conv2', torch.nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=1)),
                            ('conv3', torch.nn.Conv2d(in_channels=10, out_channels=1, kernel_size=3, padding=1)),
                            ('flatten', torch.nn.Flatten(start_dim=1)),
                            ('fc1', torch.nn.Linear(input_size, hidden_sizes[0])),
                            ('relu1', torch.nn.ReLU()),
                            ('fc2', torch.nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                            ('relu2', torch.nn.ReLU()),
                            ('fc3', torch.nn.Linear(hidden_sizes[1], hidden_sizes[2])),
                            ('relu3', torch.nn.ReLU()),
                            ('logits', torch.nn.Linear(hidden_sizes[2], output_size))]))
    
    #model = Model()
    model = model2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    data_root = '../cv-word-guesser/data/numpy_bitmap'
    batch_size = 1000
    shuffle = True
    num_workers = 0

    dataset = files.NumpyDataset(data_root, pre=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)

    for epoch in range(21):
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            preds = model.forward(inputs)
            loss_value = loss(preds, labels)
            loss_value.backward()
            optimizer.step()
            
        if epoch % 3 == 0:
            test_preds = preds.argmax(dim=1)
            print((test_preds == labels).float().mean())

    torch.save(model.state_dict(), "model_state_dict.pt")
    torch.save(model, "model.pt")

    t1 = time.time()

    print(t1-t0)

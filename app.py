import cv2
import json
import flask
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch, torchaudio
from flask import request
from moviepy.editor import *
from collections import Counter
from skimage import transform
import numpy as np
from twilio.rest import Client

app = flask.Flask(__name__)
app.config["DEBUG"] = True

mapper = pickle.load(open("index_class_map.pkl","rb"))

audio_params = {
    'input_dim': 64,
    'kernel_size_1': 32,
    'kernel_size_2': 16,
    'kernel_size_3': 8,
    'kernel_size_4': 4,
    'kernel_size_5': 2,
    'lstm_embedding_size': 798,
    'fc1_out_size': 1024,
    'fc2_out_size':1024,
    'out_size': len(mapper),
}

video_params = {
    'conv1_cin': 3,
    'conv1_cout': 96,
    'kernel_size1': 3,
    'conv2_cout': 128,
    'kernel_size2': 4,
    'conv3_cout': 32,
    'kernel_size3': 5,
    'conv_out': 25*25*32,
    'fc1_out': 128,
    'fc2_out': 64,
    'nn_out': len(mapper),
}

class VideoModel(nn.Module):
    def __init__(self, params):
        super(VideoModel, self).__init__()
        self.params = params
        self.conv1 = nn.Conv2d(params['conv1_cin'], params['conv1_cout'], params['kernel_size1'])
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(params['conv1_cout'], params['conv2_cout'], params['kernel_size2'])
        self.conv3 = nn.Conv2d(params['conv2_cout'], params['conv3_cout'], params['kernel_size3'])
        self.fc1 = nn.Linear(params['conv_out'], params['fc1_out'])
        self.fc2 = nn.Linear(params['fc1_out'], params['fc2_out'])
        self.fc3 = nn.Linear(params['fc2_out'], params['nn_out'])

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        #x - 111, 111, 96
        x = self.pool(F.relu(self.conv2(x)))
        #x - 54, 54, 128
        x = self.pool(F.relu(self.conv3(x)))
        #x - 25, 25, 32
        x = x.view(-1, 25*25*32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class AudioModel(nn.Module):

  def __init__(self, params):
    super(AudioModel,self).__init__()
    self.params = params

    self.LSTM = nn.LSTM(params['input_dim'], params['lstm_embedding_size'], num_layers = 2)

    self.fc1 = nn.Linear(params['lstm_embedding_size'], params['fc1_out_size'])
    self.fc2 = nn.Linear(params['fc1_out_size'], params['fc2_out_size'])
    self.out = nn.Linear(params['fc2_out_size'], params['out_size'])

  def forward(self, X):
    X = X.permute(2, 0, 1)
    lstm_output, lstm_state_tuple = self.LSTM(X)
    fc1_out = F.relu(self.fc1(lstm_output[-1,:,:].view(-1, self.params['lstm_embedding_size'])))
    fc2_out = F.relu(self.fc2(fc1_out))
    out = self.out(fc2_out)
    return out

def process_input2(file_path):
  vidcap = cv2.VideoCapture(file_path)
  images = []
  sec = 0
  frameRate = 0.1 #//it will capture image in each 0.5 second
  count=1
  success = getFrame(vidcap,images,sec)
  while success:
      count = count + 1
      sec = sec + frameRate
      sec = round(sec, 2)
      success = getFrame(vidcap,images,sec)
  images = np.array(images)
  idx = np.random.randint(images.shape[0], size=25)
  images = images[idx, :,:,:]
  images = np.transpose(images,(0,3,1,2))
  return torch.tensor(images, dtype=torch.float)

def getFrame(vidcap,images,sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        images.append(transform.resize(image,(224,224)))
    return hasFrames

def process_input1(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    if waveform.size(0) > 1:
        waveform = torch.unsqueeze(waveform[0,:],0)
    # m_specgram = torchaudio.transforms.MelSpectrogram(n_fft=400, sample_rate=8000, n_mels=64)(waveform)
    mfcc = torchaudio.transforms.MFCC(n_mfcc=64, sample_rate=16000, dct_type=2, log_mels=True)(waveform)
    # power = torchaudio.transforms.ComputeDeltas(win_length=5, mode='replicate')(m_specgram)
    return mfcc

def post_process_images(preds):
    preds = torch.argmax(preds,dim=1).numpy().tolist()
    cntr = dict(sorted(Counter(preds).items()))
    prob_arr = [0.]*len(mapper)
    for i in range(len(mapper)):
        if i in cntr:
            prob_arr[i] = float(cntr[i])/len(preds)
    return torch.unsqueeze(torch.tensor(prob_arr, dtype=torch.float),0)


def sms_notification(reason):
    account_sid = "AC8e25d15518b19d3d6c90b3e4efdb77bb"
    auth_token = "cfc64410364fb2e67af6b41d35b51f85"
    client = Client(account_sid, auth_token)

    message = client.messages \
                .create(
                     body=reason,
                     from_='+16466634750',
                     to='+14806850966'
                 )

@app.route('/', methods=['POST'])
def home():
    v = request.data
    with open("video.mp4","wb") as f:
        f.write(v) 
    # Saving video and audio
    video = VideoFileClip("video.mp4")
    audio = video.audio
    audio.write_audiofile('audio.wav')
    # Preprocessing input
    a_inp = process_input1("audio.wav")
    v_inp = process_input2("video.mp4")
    # Loading models
    a_model = AudioModel(audio_params)
    v_model = VideoModel(video_params)
    a_model.load_state_dict(torch.load('audio_model.pth'))
    a_model.eval()
    v_model.load_state_dict(torch.load('video_cpu_model.pth'))
    v_model.eval()
    # Get predictions
    a_out = torch.nn.functional.softmax(a_model(a_inp))
    # print("Audi Output",np.round(a_out.detach().numpy(),2))
    v_out = torch.nn.functional.softmax(v_model(v_inp))
    # Get majority voting for images
    v_out = post_process_images(v_out)
    # New probability together
    out = torch.nn.functional.softmax(torch.add(v_out,a_out))
    # Sort and get top 3 features
    prob, sort = torch.sort(out, descending=True)
    sort = [mapper[k] for k in sort[0].numpy().tolist()][:3]
    rp = np.sum(prob[0][3:].detach().numpy().tolist())/3
    prob = np.round(prob[0][:3].detach().numpy().tolist(), 2)+rp
    res={}
    for i,k in enumerate(sort):
        res[k]=prob[i]
        # if i==0 and prob[i]>0.7:
        #     sms_notification("Baby has belly pain!")
        # else:
    sms_msgs = {
        'bellypain' : "Mommm! I am in danger! MY STOMACH IS PAINING",
        'burp' : "Never mind, Just burping",
        'discomfort' : "TAKE ME OUTTA HERE Mom. Feeling uncomfortable",
        'hungry' : "FEED ME Mummmaaaaa, Sobsss!!",
        'tired' : "Time for a bed time storyyy.. I am falling asleeeeepp",
    }
    predicted_class = list(res.keys())[0]
    # print(predicted_class)
    # print(str(np.argmax(prob)))
    sms_notification(sms_msgs[predicted_class])
    return json.dumps(res)
    
app.run()
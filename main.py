import torch
import numpy as np
from PIL import Image

import clip
import wav2clip
import cv2

from load_data import Load_data
from metrics import Compute_metrics
import torchaudio
import itertools
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloader import ImageDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ", device)

#Load video files
dir_path = "C:\\Masters\\semester_3\\Project_1\\data\\vggsound_example" #ERB3_Stimuli
loading = Load_data(dir_path)
file_list = loading.get_file_list()

cos_similarities = []
top_similarity = []
img_embd = []
aud_embd = []

def batch(iterable, batch_size):
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, batch_size))
        if not batch:
            break
        yield batch

def encode_video(frames):
    """Takes a single frame and return a CLIP embedding"""
    #Load model : ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    model_vid, preprocess = clip.load("ViT-B/16") #"ViT-B/32", device=device

    #Preprocess image
    frames_inputs=[]
    for frame in frames:
        frame_preprocessed = preprocess(Image.fromarray(frame, 'RGB')).to(device)  #.unsqueeze(0)
        frames_inputs.append(frame_preprocessed)
    #frame = preprocess(Image.fromarray(frame, 'RGB')).unsqueeze(0).to(device)
    
    with torch.no_grad():
        #extract image features
        frame_embedding = model_vid.encode_image(torch.stack(frames_inputs))

    return frame_embedding

def encode_audio(audio):
    """Takes audio and return a CLIP embedding"""
    model_aud = wav2clip.get_model() #For 16Hz sample rate: (frame_length=16000, hop_length=16000)
    print("Model loaded!!!")

    with torch.no_grad():
        # model expects np.ndarray and our embedding is also a numpy array
        audio_embeddings = wav2clip.embed_audio(np.array(audio), model_aud) # (2, 512)

    return audio_embeddings


#Iterate through each batch
for batch_files in batch(file_list, batch_size=5):
    image_embeddings = []
    audio_embeddings = []

    #Iterate through files in a batch
    for file in batch_files:
        #Extract a frames and audio for every video clip
        frames = loading.extract_frames(file)
        audio = loading.extract_audio(file)

        #write audio file and load it again
        audio.write_audiofile("output_audio_file.wav") # our fps is 44100, but we can set fps=16000
        audio_data, sample_freq = torchaudio.load("./output_audio_file.wav")
        print("Video and Audio is loaded!!!")

        #Embedd frames from a video and compute mean of the embeddings
        #for i in range(0, len(frames), batch_size): #for frame in frames:
        #    batch_frames = frames[i:i+batch_size]
        frame_embedding = encode_video(frames) #frame
        #frame_embed_torch = torch.cat(frame_embedding, dim=0)
        video_mean_embedding = torch.unsqueeze(torch.mean(frame_embedding, 0), 0) #torch.Size([1, 512])
            
        #Embedd audio
        audio_embedding = encode_audio(audio_data.to(device))
        audio_embedding = (np.mean(audio_embedding, 0)).reshape(1, 512)

        #append embeddings of videos and audios for a batch
        image_embeddings.append(video_mean_embedding)
        audio_embeddings.append(audio_embedding)
    print("Embedding is complete!")

    #Compute cosine similarity between video and audio embeddings within a batch
    image_embeddings = torch.cat(image_embeddings, dim=0)
    audio_embeddings = torch.tensor(np.concatenate(audio_embeddings, 0))
    metric = Compute_metrics(image_embeddings, audio_embeddings)
    cos_similarity = metric.pair_cosine_similarity()
    logits = metric.logits()
    print("Cosine similarities: ", cos_similarity)
    print("Logits: ", logits)

    '''
    #Top similarity
    Video_mean_embedding /= Video_mean_embedding.norm(dim=-1, keepdim=True)
    audio_embeddings /= (torch.tensor(audio_embeddings)).norm(dim=-1, keepdim=True)
    similarity = (100.0 * Video_mean_embedding @ audio_embeddings.T).softmax(dim=-1) # softmax converts it to probability distribution
    values, indices = similarity[0].topk(1) #5
    '''
    
    #concatenate the scores and compute average
    #cos_similarities = torch.cat(cos_similarities, dim=0).cpu()
    #print("After concatenation: ", cos_similarities)
    #print("Average of cosine similarities: ", torch.mean(cos_similarities))
    #print("Top similarity: ", top_similarity)

'''
_MODELS = {
"RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
"RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
"RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
"RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
"RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
"ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
"ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
"ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
"ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}
'''
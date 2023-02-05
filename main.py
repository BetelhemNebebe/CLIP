import torch
import numpy as np
from PIL import Image

import clip
import wav2clip

from load_data import Load_data
from metrics import Compute_metrics
import torchaudio

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ", device)

#Load video files
dir_path = "C:\\Masters\\semester_3\\Project_1\\data\\ERB3_Stimuli"
loading = Load_data(dir_path)
file_list = loading.get_file_list()

cos_similarities = []
top_similarity = []

for file in file_list:
    #Extract a mean image and audio for every video clip
    image = loading.extract_avg_frame(file)
    audio = loading.extract_audio(file)

    #write audio file and load it again
    audio.write_audiofile("output_audio_file.wav", fps=audio.fps)
    audio_data, sample_freq = torchaudio.load("./output_audio_file.wav")
    print("Video and Audio is loaded!!!")

    #Load model
    model_vid, preprocess = clip.load("ViT-B/32", device=device)
    model_aud = wav2clip.get_model() #frame_length=16000, hop_length=16000
    print("Model loaded!!!")

    #Preprocess image
    image = preprocess(Image.fromarray(image, 'RGB')).unsqueeze(0).to(device)

    #logits_per_image, logits_per_audio = model_vid(image, audio_data)
    #probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    with torch.no_grad():
        #extract image and audio features
        image_features = model_vid.encode_image(image)
        audio_embeddings = wav2clip.embed_audio(np.array(audio_data), model_aud) # model expects np.ndarray and our embedding is also a numpy array 

    #Compute cosine similarity
    metric = Compute_metrics(image_features, audio_embeddings)
    cos_similarity = metric.cosine_similarity()

    image_features /= image_features.norm(dim=-1, keepdim=True)
    audio_embeddings /= (torch.tensor(audio_embeddings)).norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ audio_embeddings.T).softmax(dim=-1) # softmax converts it to probability distribution
    values, indices = similarity[0].topk(1) #5

    #Append similarity values
    cos_similarities.append(cos_similarity)
    top_similarity.append(values)

print("Embedding is complete!")

print("Cosine similarities: ", cos_similarities)
#concatenate the scores and compute average
cos_similarities = torch.cat(cos_similarities, dim=0).cpu()
print("After concatenation: ", cos_similarities)
print("Average of cosine similarities: ", torch.mean(cos_similarities))
print("Top similarity: ", top_similarity)
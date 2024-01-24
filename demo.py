import os
from typing import *
import gdown
from PIL import Image, ImageOps

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

import streamlit as st

st.set_page_config(
    page_title='koVAR demo',
    layout = 'wide',
    initial_sidebar_state='expanded',
)
st.markdown("""<style>
.option-text {
        font-size:18px;
        font-family:Sans-serif;
        font-weight: bold;
            }
.option-border{
        border:1px solid #dee2e6;
        border-radius: 10px;
        padding:1rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model():
    with st.status("Please wait ...", expanded=True) as status:
        if not os.path.exists(st.secrets.model.output):
            st.write("Downloading model...")
            gdown.download(id=st.secrets.model.file_id, output=st.secrets.model.output, quiet=False)
        st.write("Loading model...")
        baseline_model = "koclip/koclip-base-pt"
        model = AutoModel.from_pretrained(baseline_model)
        processor = AutoProcessor.from_pretrained(baseline_model)
        model.load_state_dict(torch.load(st.secrets.model.output, map_location='cpu')['model_state_dict'])
        status.update(label="Complete!", state="complete", expanded=False)
        return model, processor
    
def show_image(image:Image):
    image = ImageOps.exif_transpose(image)
    st.image(image)

def format_text(o2, hyp):
    return f'{hyp} {o2}'

def show_details(probs=torch.tensor):
    with st.expander('See details'):
        option1, option2 = st.columns(2)
        option1.metric(label = 'Option1', value=f"{probs[0]:.4f}%")
        option2.metric(label = 'Option2', value=f"{probs[1]:.4f}%")


if __name__ == '__main__':
    # Title
    st.title('Korean Visual Abductive Reasning')

    # Side-bar ( user input )
    with st.sidebar:
        example_on = st.toggle('See Example', value=False)
        # Observation
        st.header(':mag: Observations')
        with st.container(border=True):
            st.write('O1')
            image_file = st.file_uploader(label = '이미지를 선택해 주세요.')
            
        
        with st.container(border=True):
            o2 = st.text_input("O2", 
                            placeholder="사진의 장면 후에 관찰된 내용을 입력하세요.",
                            value = '여자는 한 손으로 핸드폰을 보면서 편하게 걸어갈 수 있었다.' if example_on else ''
                            )
        # Hypothesis
        st.header(':face_with_monocle: Hypothesis')
        with st.container(border=True,):
            h1 = st.text_input('H1', placeholder="첫 번째 가설을 작성하세요.",
                               value = "여자는 짐을 길바닥에 버렸다." if example_on else '')
            h2 = st.text_input('H2', placeholder="두 번째 가설을 작성하세요.",
                               value="같이 온 남자가 여자의 짐을 같이 들어주었다." if example_on else '')
        # Button
        submit = st.button('submit')

    # Main
    section1, section2 = st.columns([0.5, 0.5], gap='large')
    with section1 :
        st.subheader('Option1')
        with section1.container(border = True):
            st.markdown(f'<p class="option-text">O1: </p>', unsafe_allow_html=True)
            if image_file is not None:
                image = Image.open(image_file)
                show_image(image)
            elif example_on:
                show_image(Image.open('./src/example.jpg'))
            else :
                show_image(Image.open('./src/default_image.png'))

            st.markdown(f'<p class="option-text option-border">H1: {h1}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="option-text option-border">O2: {o2}</p>', unsafe_allow_html=True)
        
        
    with section2 :
        st.subheader('Option2')
        with section2.container(border = True):
            st.markdown(f'<p class="option-text">O1: </p>', unsafe_allow_html=True)
            if image_file is not None:
                image = Image.open(image_file)
                show_image(image)
            elif example_on:
                image = Image.open('./src/example.jpg')
                show_image(image)
            else :
                image=None
                show_image(Image.open('./src/default_image.png'))
            st.markdown(f'<p class="option-text option-border">H1: {h2}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="option-text option-border">O2: {o2}</p>', unsafe_allow_html=True)
    st.divider()
    
    # Results
    if submit:
        st.header('Results')
        if image and o2 and h1 and  h2:
           
            model, processor = load_model()
            input_texts = [format_text(o2, h1), format_text(o2, h2)]
            inputs = processor(
                images=image, 
                text=input_texts, 
                return_tensors='pt', 
                padding=True )
            
            result = model(**inputs)
            probs = F.softmax(result.logits_per_image)
        
            answer = 'Option 1' if probs[0][0]>probs[0][1] else 'Option 2'
            
            st.subheader(f":white_check_mark: '{answer}' is more plausible!")
            show_details(probs[0]*100)
        else:
            st.error('이미지와 텍스트를 모두 채워주세요', icon="⚠️")
        
        

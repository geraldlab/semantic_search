import streamlit as st

from collections import Counter
import nltk
from nltk.corpus import stopwords

import torch
from datasets import load_dataset

import time
import sys, os
import logging

#custom packages
sys.path.insert(1, os.getcwd())

from src import constant as my_constant

from src import my_utils as my_utils
from src import searcher as my_searcher

st.set_page_config(layout="wide")

st.title('Demo of Semantic Search on United Nations Administrative Instructions (AIs)')
st.markdown(f"{my_constant.open_i}- Data: web scraped from UN Policy portal: https://policy.un.org/browse-by-source/30776{my_constant.close_i}")
st.markdown(f"{my_constant.open_i}- Technology used: Sentence transformer model, FAISS (Facebook AI Similarity Search), YAKE (unsupervised model), Huggingface arrow dataset, and Selenium (dynamic web page scraping){my_constant.close_i}")

#get configuration
cfg = my_utils.get_configuration()

search_cfg=cfg[my_constant.search_setting] 


log_dir = cfg.get('log_dir')

#search config
search_cfg = cfg[my_constant.search_setting] 
max_len = search_cfg.get(my_constant.max_doc_len) if search_cfg.get(my_constant.max_doc_len) else 800


#config device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_data_model():
    try:
        #load data
        search_ds_path = os.path.join(os.getcwd(), "data")
       
        #load from disk
        search_dataset = load_dataset('parquet', data_files=os.path.join(search_ds_path, 'embed_dataset.parquet'), split="train")
       
        if search_dataset is None:
            st.write("Ops sorry! failed to load data")
            raise Exception("Failed to load dataset!!")

        #add faiss index
        search_dataset.add_faiss_index(column=my_constant.embeddings)
        
        nltk.download('stopwords')
        time.sleep(.1)
        #load stop words
        stop_words = stopwords.words('english')

        st_wd = search_cfg.get(my_constant.stop_words)
        if st_wd:
            stop_words = stop_words + [str(s).strip().lower() for s in st_wd.split(my_constant.comma) if s]
    
        st.write(search_dataset)
                 
        model_path = os.path.join(os.getcwd(), 'model')
        model_path = os.path.join(model_path, 'multi-qa-mpnet-base-dot-v1')

        #loading model
        sentence_tokenizer, sentence_model = my_searcher.load_sentence_model_tokenizer(model_path, device)
    
        if sentence_model is None:
            st.write(my_constant.abort_msg )
            raise Exception(f'failed to load model from: {model_path}')

        return {
                'search_dataset': search_dataset, 
                'stop_words': Counter(stop_words), 
                'sentence_tokenizer': sentence_tokenizer,
                'sentence_model': sentence_model,
                'device': device
                }
    
    except Exception as e:
        logging.error(f'Home.load_data_model: {str(e)}')

searcher_dict = load_data_model()

#info
st.write(searcher_dict['search_dataset'])


try:
    with st.form('Search'):
        search_for = st.text_input('Search for:')
        num_recs = st.slider('Show only Top: ', min_value=1, max_value=50, value=20)

        submit = st.form_submit_button('Search')

    if submit:#run the search
        
        results, time_tkn = my_searcher.search_for_documents(search_for, 
                                                             searcher_dict, 
                                                             prev_len=200, k=num_recs)
    
 
        
        st.markdown(f"{my_constant.open_i}Search took:{time_tkn}.{my_constant.close_i}")
                   
       
        if len(results) > 0:
            my_searcher.print_streamlit_results(results)
        else:
            st.markdown(f'{my_constant.opening_tag}No documents found with specified critera.{my_constant.closing_tag}')
       
        
        st.markdown(f"{my_constant.open_i}{my_constant.score_defn}{my_constant.close_i}")


except Exception as e:
    logging.error(f'{str(e)}')  
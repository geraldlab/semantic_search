import pandas as pd

from collections import defaultdict

from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModel

import os
from pathlib import Path
from datetime import datetime
import time
import streamlit as st

import logging


#-- custom module
from src import constant as my_constant
from src import my_utils as my_utils


column_to_index = my_constant.text

printable_cols = [my_constant.title, 'st_ai', my_constant.content, my_constant.keywords, 
                      my_constant.url, my_constant.date_created,
                        my_constant.scores]

'''
###########################################################
    -  create embedding
'''

def get_embeddings(text_list, sentence_tokenizer, sentence_model, device):
    encoded_input = sentence_tokenizer(
            text_list, padding=True, truncation=True, return_tensors="pt"
        )
    
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = sentence_model(**encoded_input)
    
    def cls_pooling(model_output):
        return model_output.last_hidden_state[:, 0]
    
    return cls_pooling(model_output)

'''
############################################################
    - Add FAISS index
'''
def add_faiss_index(content_dataset, file_name=None):
    #add faiss index
    content_dataset.add_faiss_index(column=my_constant.embeddings)

    if file_name:
       content_dataset.save_faiss_index(my_constant.embeddings, file_name)


'''
#############################################################
    - Search 
'''

def add_highlight_markers(search, content, stop_words, prev_len):
   '''
    - method inserts open and closing tags, to for search in content
   '''
 
   kw = set(my_utils.remove_punctuation(my_utils.remove_stopwords(search, stop_words)).split())
   val = set(my_utils.remove_punctuation(my_utils.remove_stopwords(content, stop_words)).split())
   
   isect_words = kw.intersection(val)
   
   num_found = 0
   if len(isect_words) > 0:
    wd_content = {}
          
    for k in list(isect_words):
        for p in my_utils.find_all_loc(k, content):
           wd_content[p] =k
    
    num_found = len(wd_content)
    #insert markers in text
    tagged_txt = ''
    start = 0

    if len(wd_content) > 0:
        wd_content = sorted(wd_content.items())
        for wd in wd_content:
            if wd[0] < prev_len + min(wd_content)[0]:
                tagged_txt = tagged_txt + content[start:wd[0]]  + my_constant.opening_tag + content[wd[0]: wd[0] + len(wd[1])] + my_constant.closing_tag
                start = wd[0] + len(wd[1])
        #update prev len
        prev_len = len(tagged_txt) + 10      
        content = tagged_txt + content[len(tagged_txt): 10]
    
   return content[:prev_len], num_found

def post_process_result(df, search_term, searcher_dict, prev_len = 200):
    h_dict = defaultdict(list)
   
    for i, row in df.iterrows():
        
        #highlight content
        num_in_content_score = 0
        _content, num_words_in_content = add_highlight_markers(search_term, row[my_constant.content], 
                                              searcher_dict['stop_words'], prev_len)
                     
        #highlight keywords
       
        _kword, num_in_kword_score= add_highlight_markers(search_term, row[my_constant.keywords], 
                                         searcher_dict['stop_words'], prev_len)

        #add to score
        num_in_kword_score = num_in_kword_score * 5
         
        #add content to dict
        h_dict[my_constant.title].append(row[my_constant.title])
        h_dict['st_ai'].append(row['st_ai'])
        h_dict[my_constant.url].append(row[my_constant.url])
        h_dict[my_constant.file_name].append(row[my_constant.file_name])
        h_dict[my_constant.content].append(_content)
        h_dict[my_constant.num_words_in_content].append(num_words_in_content)

        h_dict[my_constant.keywords].append(_kword)
        h_dict['num_in_kword_score'].append(num_in_kword_score)

        score =  row[my_constant.scores] + num_in_content_score + num_in_kword_score
        
        h_dict[my_constant.scores].append(f'{score:.1f}')

        h_dict[my_constant.date_created].append(row[my_constant.date_created])


    df = pd.DataFrame(h_dict)
    if len(df) > 0:
       df[my_constant.scores] = pd.to_numeric(df[my_constant.scores], errors='ignore')
 
    
    return df


def get_search_result(search_dataset, search_term, 
                      sentence_tokenizer, sentence_model, device, k=5):
    
    
    search_embedding =  get_embeddings([search_term], sentence_tokenizer, sentence_model, device).cpu().detach().numpy()
    st.write(f'Searched for:*{search_term}')
    #run the search
    scores, samples = search_dataset.get_nearest_examples(my_constant.embeddings, search_embedding, k=k )
    st.write(f'Scores:{scores} !!')
    results = pd.DataFrame.from_dict(samples)

    results[my_constant.scores] = scores
    results = results.sort_values([my_constant.scores, 'dte0'], ascending=[False, True])

    return results
    


def search_for_documents(search_for, searcher_dict, prev_len, k=10):
    start_tme = datetime.now()
    
    results = get_search_result(searcher_dict['search_dataset'], search_for, 
                                              searcher_dict['sentence_tokenizer'], 
                                              searcher_dict['sentence_model'], 
                                              searcher_dict['device'], k=k)

    st.write(f'results are{len(results)} - {results.columns}')
    
    results = results.drop_duplicates(subset=printable_cols)

    marked_result = post_process_result(results, search_for, searcher_dict, prev_len=prev_len)

    return marked_result, my_utils.get_time_taken(start_tme)

'''
###########################################################################
    - Load content database    
'''
def get_faiss_idx_path(working_dir):
    
    faiss_idx = None
    for _f in os.listdir(working_dir):
        fp = os.path.join(working_dir, _f)
        p = Path(fp)

        if os.path.exists(fp) and p.is_file() and p.name.endswith(my_constant.faiss):
            return fp
        
      

'''
###########################################################################
    - Visualize Results
'''

def print_streamlit_results(_df):
       
       for i, row in _df.iterrows():
        st.markdown(f"{my_constant.opening_tag}{row[my_constant.title]}{my_constant.closing_tag}")
        st.write(f"Symbol: {row['st_ai']} [link]({row[my_constant.url]})")
        content = my_utils.remove_multiple_spaces(row[my_constant.content].strip())

        st.markdown(f"{my_constant.open_i}{content}{my_constant.close_i} ...")

        st.markdown(f"{my_constant.open_i}Keywords:{row[my_constant.keywords]}{my_constant.close_i}")
        st.markdown(f"{my_constant.open_i}Number of mentions  of search word(s) in document : {my_constant.opening_tag}{row[my_constant.num_words_in_content]}{my_constant.closing_tag}{my_constant.close_i}")
        st.markdown(f"{my_constant.opening_tag}{row[my_constant.url]}{my_constant.closing_tag}")
       
        st.markdown(f'{my_constant.open_i}Publication Date: {row[my_constant.date_created]}{my_constant.close_i}') 
        
        st.markdown(f'{my_constant.open_i}*score: {row[my_constant.scores]:.1f}{my_constant.close_i}')
        
        st.write("-" * 100)

      
    
'''
#############################################################################
    - Load model and tokenizer
'''
def load_sentence_model_tokenizer(sentence_model_path, device):
    #initialise model
    sentence_model_path = os.path.join(os.getcwd(), sentence_model_path)

    sentence_tokenizer = AutoTokenizer.from_pretrained(sentence_model_path)
    sentence_model = AutoModel.from_pretrained(sentence_model_path)

    sentence_model.to(device)

    return sentence_tokenizer, sentence_model
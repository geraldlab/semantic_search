import pandas as pd

from collections import defaultdict


import os, re
from pathlib import Path
from datetime import datetime
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
#############################################################
    - Search 
'''

def add_highlight_markers(search_for, content, stop_words):
   
    #get common words on both content
    kw = set(my_utils.remove_punctuation(my_utils.remove_stopwords(str.lower(search_for), stop_words)).split())
    val = set(my_utils.remove_punctuation(my_utils.remove_stopwords(str.lower(content), stop_words)).split())
   
    #common words
    def replace_all(pattern, content) -> str:
        c_wds = re.findall(pattern, content, re.IGNORECASE)
        i = len(c_wds)
        occurences = list(set(c_wds))
                 
        for occurence in occurences:
            repl = my_constant.opening_tag + occurence.strip() + my_constant.closing_tag
            content = content.replace(occurence, repl)

        return content, i
    
    x = 0
    for wd in kw.intersection(val):
        content, i = replace_all(wd, content)
        x = x + i
   
    return content, x

def post_process_result(df, search_term, searcher_dict):
    h_dict = defaultdict(list)
   
    for i, row in df.iterrows():
        
        #highlight content
        #highlight content
        num_words_in_content = 0
        _content = row[my_constant.content]
       

        if _content and _content.strip() != '':
            _content, num_words_in_content = add_highlight_markers(search_term, _content, 
                                              searcher_dict['stop_words'])
                     
        #highlight keywords
        _kword = row[my_constant.keywords]
        num_in_kword_score = 0

        if _kword and _kword.strip() != '':
            _kword, num_in_kword_score= add_highlight_markers(search_term, _kword, 
                                         searcher_dict['stop_words'])

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

        score =  row[my_constant.scores] + num_in_kword_score
        
        h_dict[my_constant.scores].append(f'{score:.1f}')

        h_dict[my_constant.date_created].append(row[my_constant.date_created])


    df = pd.DataFrame(h_dict)
    if len(df) > 0:
       df[my_constant.scores] = pd.to_numeric(df[my_constant.scores], errors='ignore')
 
    
    return df


def get_search_result(search_dataset, search_term, 
                      sentence_tokenizer, sentence_model, device, k=5):
    
    
    search_embedding =  get_embeddings([search_term], sentence_tokenizer, sentence_model, device).cpu().detach().numpy()
   
    #run the search
    scores, samples = search_dataset.get_nearest_examples(my_constant.embeddings, search_embedding, k=k )

    results = pd.DataFrame.from_dict(samples)

    results[my_constant.scores] = scores
    results = results.sort_values([my_constant.scores, 'dte0'], ascending=[False, True])

    return results
    


def search_for_documents(search_for, searcher_dict, k=10):
    try:
        start_tme = datetime.now()
    
        results = get_search_result(searcher_dict['search_dataset'], search_for, 
                                              searcher_dict['sentence_tokenizer'], 
                                              searcher_dict['sentence_model'], 
                                              searcher_dict['device'], k=k)

     
        results = results.drop_duplicates(subset=printable_cols)

        marked_result = post_process_result(results, search_for, searcher_dict)

        return marked_result, my_utils.get_time_taken(start_tme)
    except Exception as e:
        logging.error(f'{str(e)}')

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
       prev_len =1300
       for i, row in _df.iterrows():
        st.markdown(f"{my_constant.opening_tag}{row[my_constant.title]}{my_constant.closing_tag}")
        st.write(f"Symbol: {row['st_ai']} [link]({row[my_constant.url]})")
        content = my_utils.remove_multiple_spaces(row[my_constant.content].strip()[:prev_len])

        st.markdown(f"{my_constant.open_i}{content}{my_constant.close_i} ...")

        st.markdown(f"{my_constant.open_i}Keywords:{row[my_constant.keywords]}{my_constant.close_i}")
        st.markdown(f"{my_constant.open_i}Number of mentions  of search word(s) in document : {my_constant.opening_tag}{row[my_constant.num_words_in_content]}{my_constant.closing_tag}{my_constant.close_i}")
        st.markdown(f"{my_constant.opening_tag}{row[my_constant.url]}{my_constant.closing_tag}")
       
        st.markdown(f'{my_constant.open_i}Publication Date: {row[my_constant.date_created]}{my_constant.close_i}') 
        
        st.markdown(f'{my_constant.open_i}*score: {row[my_constant.scores]:.1f}{my_constant.close_i}')
        
        st.write("-" * 100)
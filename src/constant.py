'''
################################################
    - CONSTANTS
'''

temp = 'temp'
log = 'log'
asset = 'asset'


utf_8 = 'utf-8'
comma=','

parquet = 'parquet'
faiss = 'faiss'

doc='.doc'
docx='.docx'
word_ext = [doc, docx]
pdf = '.pdf'

file_types = {
             
              word_ext[0]:word_ext[0],
              word_ext[1]:word_ext[0],

              pdf: pdf,
              f'.{parquet}': parquet,
              f'.{faiss}': faiss,
             }


#fields for dataset
title='title'
url = 'url'
file_name = 'file_name'
embeddings = "embeddings"
content = 'content'
keywords = 'keywords'
text = 'text'
num_words_in_content= 'num_words_in_content'
date_created = 'date_created'
scores = 'scores'
extension='extension'


opening_tag = '**:blue['
closing_tag = ']**'
open_i='_' #italic open markdown
close_i='_' #italic close markdown

embed_ds_pfx = 'embed_ds'

#yaml file keys
yaml_file_name='search_documents_yaml.yaml'
app='APP'
search_setting= 'SEARCHER_SETTINGS'

model_path='MODEL_RELATIVE_PATH'
searcher_settings = 'SEARCHER_SETTINGS'
asset_dir ='ASSETS_DIR'

file_extension_to_idex = 'FILE_EXTENSIONS_TO_INDEX'
file_extension_to_ignore = 'FILE_EXTENSIONS_TO_IGNORE'
replace_user_profile = 'REPLACE_USER_PROFILE_PATH'

num_recs_return = 'DEFAULT_NUMBER_OF_RECORDS_RETURNED_BY_SEARCH'
max_preview = 'PREVIEW_LENGTH'
max_doc_len = 'MAX_DOCUMENT_LENGTH'
stop_words = 'STOP_WORDS'

abort_msg = f'Ops! Sorry something went wrong !!'
score_defn= '*Score - is the similarity metric calculated by Semantic search.'
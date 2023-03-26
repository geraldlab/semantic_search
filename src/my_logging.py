import logging
import os
from datetime import datetime

from src import constant as my_constant

log_dir = os.path.join(os.getcwd(), my_constant.log )

log_fname = os.path.join(log_dir, f'search_documents_{datetime.now().strftime("%d_%m_%Y")}.log')

if not os.path.exists(log_fname):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(log_fname, 'w'):
        os.utime(log_fname, None)

logging.basicConfig(filename = log_fname,
                    level=logging.DEBUG,
                    format = '%(asctime)s:%(levelname)s:%(message)s',
                    force=True)
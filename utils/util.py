import logging

def set_logging(log_path):
    logger = logging.getLogger()
    logger.setLevel('INFO')
    BASIC_FORMAT = '%(levelname)s: %(message)s'
    # DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT)
    chlr = logging.StreamHandler() # 输出到控制台的handler
    chlr.setFormatter(formatter)
    # chlr.setLevel('INFO')  # 也可以不设置，不设置就默认用logger的level
    fhlr = logging.FileHandler(log_path) # 输出到文件的handler
    fhlr.setFormatter(formatter)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    
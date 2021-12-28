import  logging

def get_logger(path):
    logger = logging.getLogger('npa')
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler('./log_'+path+'.txt','w')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(message)s ')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

def print_args(args,logger):
    output = ''
    for arg, content in args.__dict__.items():
        output=output+"{}:{} ;".format(arg, content)
    logger.info(output)
import argparse
from papers_collect import cvpr

def main(args):
    """main function
    Args:
        args argparse.Namespace: Configs
    """
    if args.conf == 'cvpr':
        collecter = cvpr.CVPR_papers_collecter(year=args.year, data_path=args.data_path)
    else:
        raise NotImplementedError

    collecter.collect()
    collecter.save_pickle()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--conf', default='cvpr')
    argparser.add_argument('--year', default='2023')
    argparser.add_argument('--data_path', default='./papers_info'')
    args = argparser.parse_args()
    main(args)

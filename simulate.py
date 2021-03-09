import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument(...)

    args = parser.parse_args()
    return args


def main(args):
    print("Hello world!")


if __name__ == '__main__':
    args = parse_args()
    main(args)
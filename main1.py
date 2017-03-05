import bz2


def read_bz2(file_path):
    """read file directly from bz2 file, return tuple of list of y and list of X
    :type file_path: str
    """
    all_X = []
    all_y = []
    with bz2.open(file_path, 'r') as f:
        for line in f:
            y, X = parse_line(line.decode('utf-8'))
            all_X.append(X)
            all_y.append(y)
    return (all_y, all_X)


def parse_line(line):
    r"""parse a single line, return tuple (y, X)
    :type line: str

    :type y: int, 0 or 1
    :type X: dict, number of occurances of each feature
    >>> parse_line('0\ta,a,a\n')
    (0, {'a': 3})
    """
    # get y
    if ((type(line.split()[0]) == 'str')):
        y = int(line.split()[0])
        X_list = line.split()[1].split(',')
     X = {feature: X_list.count(feature) for feature in set(X_list)}
        return (y, X)
    else:
        y = None
        X_list = line.split()[0].split(',')
    # count occurances of each unique feature in line
        X = {feature: X_list.count(feature) for feature in set(X_list)}
        return (y, X)




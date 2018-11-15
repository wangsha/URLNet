import string
import numpy as np
import pandas as pd

try:
    from urllib.parse import urlparse, unquote
except ImportError:
    from urlparse import urlparse, unquote


def entropy2(labels, base=None):
    """ Computes entropy of label distribution. """
    from math import log, e

    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    # Compute entropy
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)

    return ent


def char_feature_expand(df):
    for i in string.ascii_lowercase + string.ascii_uppercase + string.digits:
        df['count_' + i] = df['url'].apply(lambda url: float(url.count(i)))


def specialchar_feature_expand(df):
    for char in "_.~!*'();:@&=+$,/\?#[%-]":
        df['count_%s' % char] = df['url'].apply(lambda url: float(unquote(url).count(char)))
        double_char = '%s%s' % (char, char)
        df['double_%s' % double_char] = df['url'].apply(lambda url: float(unquote(url).count(
            double_char)))


def other_feature_expand(df):
    def _count_special_words(url):
        url = url.lower()
        count = 0
        for word in ['.exe', '.zip', '.dll', 'signin', 'password', 'reset', 'login', 'buy',
                     'pay', 'download', 'confirm']:
            count += url.count(word)
        return count
    df['special_word'] = df['url'].apply(_count_special_words)


def tokenize_row(row):
    tokens = [row['scheme']]
    tokens += row['hostname'].split('.')
    tokens += row['path'].split('/')
    tokens += row['query'].split('=')
    return tokens


def urlparse_feature_expand(df):
    import re

    def _parse_url(url):
        if not (url.startswith('http://') or url.startswith('https://')):
            url = 'http://%s' % url
        x = urlparse(url)

        hostname = x.hostname if x.hostname else ""

        not_ip = 1
        try:
            if re.match('\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', hostname):
                not_ip = 0
        except Exception as e:
            print(url)
            print(x)

        primary_domain, top_level_domain, subdomain = '', '', ''
        if not_ip and hostname.count(".") > 1:
            primary_domain = hostname[hostname.find('.') + 1:]
            top_level_domain = hostname.split('.')[-1]
            sub_domain = hostname.split('.')[0]
        filename = ''
        if not_ip and x.path:
            token = x.path.split('/')[-1]
            if token.count('.') > 0:
                filename = token

        port = 0
        try:
            port = x.port
        except Exception as e:
            print(url)
            print(x)

        return x.scheme, hostname, x.path, x.params, x.query, x.fragment, port, hostname, \
               primary_domain, top_level_domain, subdomain, filename, not_ip

    df['scheme'], df['domain'], df['path'], df['params'], df['query'], df['fragment'], df['port'], \
    df['hostname'], df['primary_domain'], df['top_level_domain'], df['subdomain'], df['filename'], \
    df['not_ip_address'] = \
        zip(*df['url'].map(_parse_url))

    for attribute in ['url', 'scheme', 'domain', 'path', 'params', 'query', 'fragment', 'hostname',
                      'primary_domain', 'top_level_domain', 'subdomain', 'filename']:
        df['len_%s' % attribute] = df[attribute].apply(lambda x: float(len(x)))
        df['entropy_%s' % attribute] = df[attribute].apply(lambda x: entropy2(list(x)))

    df['domain_contain_number'] = df['hostname'].apply(lambda s: int(any(i.isdigit() for i in s)))
    df['ratio_hostname_url'] = df['len_hostname'] / df['len_url']
    df['ratio_subdomain_hostname'] = df['len_subdomain'] / df['len_hostname']
    df['tokens'] = df.apply(tokenize_row, axis=1)
    df['count_tokens'] = df['tokens'].apply(len)
    df['max_token_len'] = df['tokens'].apply(lambda lst: max(map(len, lst)))
    df['mean_token_len'] = df['tokens'].apply(lambda lst: sum(map(len, lst)) / len(lst))
    df['ration_max_token_len_url_len'] = df['max_token_len'] / df['len_url']


def normalize_dataframe(df):
    from sklearn import preprocessing
    def normalize_dataframe(df):
        # Create a minimum and maximum processor object
        mms = preprocessing.MinMaxScaler()

        # Create an object to transform the data to fit minmax processor
        x_scaled = mms.fit_transform(df)
        df_normalized = pd.DataFrame(x_scaled)

        return df_normalized

    # Create a minimum and maximum processor object
    mms = preprocessing.MinMaxScaler()

    # Create an object to transform the data to fit minmax processor
    x_scaled = mms.fit_transform(df)
    df_normalized = pd.DataFrame(x_scaled)

    return df_normalized


def get_expert_features(urls):
    df = pd.DataFrame(urls, columns=['url'])
    size1 = df.shape[1]
    char_feature_expand(df)
    size2 = df.shape[1]
    specialchar_feature_expand(df)
    size3 = df.shape[1]
    urlparse_feature_expand(df)
    size4 = df.shape[1]
    other_feature_expand(df)
    size5 = df.shape[1]
    df.to_csv("temp/url_features_%s.csv" % len(urls))
    df = df._get_numeric_data().replace(np.nan, 0)
    size6 = df.shape[1]
    df = normalize_dataframe(df)
    return df.values

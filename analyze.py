import os
import sys
import json
import math
import spacy
import pickle
import numpy as np
import networkx as nx

from IPython import embed

from tqdm import tqdm
from numpy.linalg import norm
from itertools import combinations_with_replacement as comb
from markdown_it import MarkdownIt
from mdit_plain.renderer import RendererPlain
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

if sys.version_info < (3, 7):
    raise RuntimError('This program requires Python 3.7 or later')

NSENTS = 4

NTERMS = 32

NCLUSTERS = 8

NROWS = 16

PCLUSTERS = 1 / 1

# make a normalizing decorator
def _normalize(func):

    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        total = sum(result)
        return [x / total for x in result]

    return wrapper


def metrics(D):

    @_normalize
    def intensity(orgs):
        return [sum(D.edges[e]['weight'] for e in D.in_edges(n)) for n in orgs]

    @_normalize
    def popularity(orgs):
        return [
            sum(math.sqrt(D.edges[e]['weight']) for e in D.in_edges(n))
            for n in orgs
        ]

    @_normalize
    def connectivity(orgs):

        G = nx.Graph()

        for x, y in comb(orgs, 2):
            G.add_edge(x, y, weight=0)

        for p in [n for n in D.nodes if D.nodes[n]['type'] != 'organization']:
            for x, y in comb([o for _, o in D.out_edges(p)], 2):
                G.edges[(x, y)]['weight'] += math.sqrt(
                    D.edges[(p, x)]['weight'] * D.edges[(p, y)]['weight'])

        result = nx.centrality.eigenvector_centrality(G, weight='weight')
        return [result[x] for x in orgs]

    orgs = sorted([n for n in D.nodes if D.nodes[n]['type'] == 'organization'])

    return {
        x: {
            'intensity': a,
            'popularity': b,
            'connectivity': c,
        }
        for x, a, b, c in zip(
            orgs,
            intensity(orgs),
            popularity(orgs),
            connectivity(orgs),
        )
    }


def models(org_shares, target, training):
    parser = MarkdownIt(renderer_cls=RendererPlain)
    processor = spacy.load('en_core_web_lg')
    embedder = SentenceTransformer('all-mpnet-base-v2')
    clusterer = KMeans(n_clusters=NCLUSTERS, random_state=0)
    summarizer = TfidfVectorizer()

    print('Processing training sentences')
    sents_training = {
        k: v
        for k, v in {
            k: [x.text.strip() for x in processor(v).sents]
            for k, v in tqdm(training.items())
        }.items() if len(v) >= NSENTS
    }

    print('Processing target sentences')
    sents_target = {
        k: [
            x.text.strip()
            for x in processor(parser.render(target[str(k)])).sents
        ]
        for k in tqdm(org_shares.keys())
    }

    print('Processing training vectors')
    path = os.path.join('cache', 'vecs-training.pickle')
    if os.path.exists(path):
        with open(path, 'rb') as fd:
            vecs_training = pickle.load(fd)
    else:
        vecs_training = {
            k: np.mean(embedder.encode(v), axis=0)
            for k, v in tqdm(sents_training.items())
        }
        with open(path, 'wb') as fd:
            pickle.dump(vecs_training, fd)

    print('Processing target vectors')
    path = os.path.join('cache', 'vecs-target.pickle')
    if os.path.exists(path):
        with open(path, 'rb') as fd:
            vecs_target = pickle.load(fd)
    else:
        vecs_target = {
            k: np.mean(embedder.encode(v), axis=0)
            for k, v in tqdm(sents_target.items())
        }
        with open(path, 'wb') as fd:
            pickle.dump(vecs_target, fd)

    print('Processing clusters')
    while True:
        clusterer.fit(list(vecs_training.values()))

        clusters = {i: [] for i in range(NCLUSTERS)}

        for c, k in zip(clusterer.labels_, vecs_training.keys()):
            clusters[c].append(k)

        if all(
                len(x) / len(vecs_training) > PCLUSTERS
                for x in clusters.values()):
            break
        else:
            outliers = set(
                sorted(
                    clusters.items(),
                    key=lambda x: len(x[1]),
                )[0][1])
            vecs_training = {
                k: v
                for k, v in vecs_training.items() if k not in outliers
            }
            clusterer = KMeans(n_clusters=NCLUSTERS, random_state=0)

    print('Processing similarities')
    similarities = {
        k: [
            np.dot(v, x) / (norm(v) * norm(x))
            for x in clusterer.cluster_centers_
        ]
        for k, v in tqdm(vecs_target.items())
    }

    components = {
        k: [float(x / sum(v)) for x in v]
        for k, v in tqdm(similarities.items())
    }

    with open('cache/components.json', 'w') as fd:
        json.dump(components, fd)

    results = {
        c: {
            'intensity': 0,
            'popularity': 0,
            'connectivity': 0
        }
        for c in range(NCLUSTERS)
    }

    for o, s in org_shares.items():
        for c in range(NCLUSTERS):
            for k, v in s.items():
                results[c][k] += v * components[o][c]

    print('Processing frequencies')
    matrix = summarizer.fit_transform(
        ' '.join([t.lemma_ for d in v for t in processor(d) if not t.is_stop])
        for v in clusters.values()).toarray()
    print(matrix.shape)
    features = summarizer.get_feature_names_out()

    # embed()

    summaries = dict(
        zip(
            clusters.keys(),
            [
                list(
                    map(
                        lambda x: x[1],
                        sorted(
                            [(matrix[i][j], features[j])
                             for j in range(matrix.shape[1])],
                            reverse=True,
                        )[:NTERMS])) for i in tqdm(range(matrix.shape[0]))
            ],
        ))

    cluster_descriptions = {
        k: {
            'count': len(v),
            'members': v,
            'tfidf': summaries[k],
        }
        for k, v in clusters.items()
    }

    return results, cluster_descriptions


def discussion(D):
    amount_organizations = [
        sum(D.edges[e]['weight'] for e in D.in_edges(n)) for n in D.nodes
        if D.nodes[n]['type'] == 'organization'
    ]

    amount_people = [
        sum(D.edges[e]['weight'] for e in D.out_edges(n)) for n in D.nodes
        if D.nodes[n]['type'] != 'organization'
    ]

    return {
        'gini-organizations':
        0.5 *
        np.abs(np.subtract.outer(
            amount_organizations,
            amount_organizations,
        )).mean() / np.mean(amount_organizations),
        'gini-people':
        0.5 * np.abs(np.subtract.outer(
            amount_people,
            amount_people,
        )).mean() / np.mean(amount_people)
    }


if __name__ == '__main__':
    with open('data/network.json') as fd:
        D = nx.node_link_graph(json.load(fd))

    with open('data/descriptions-ibis.json') as fd:
        descriptions_ibis = json.load(fd)

    with open('data/descriptions-all.json') as fd:
        descriptions_all = json.load(fd)

    org_shares = metrics(D)

    cls_shares, clusters = models(
        org_shares,
        descriptions_ibis,
        descriptions_all,
    )

    extra = discussion(D)

    with open('data/org-shares.json', 'w') as fd:
        json.dump(org_shares, fd, indent=2)

    with open('data/cls-shares.json', 'w') as fd:
        json.dump(cls_shares, fd, indent=2)

    with open('data/clusters.json', 'w') as fd:
        json.dump(clusters, fd, indent=2)

    with open('data/extra.json', 'w') as fd:
        json.dump(extra, fd, indent=2)

    for property in ['intensity', 'popularity', 'connectivity']:
        with open(f'data/org-{property}.tex', 'w') as fd:
            fd.write('\n'.join([
                '{} & {} & {} \\\\'.format(
                    i + 1,
                    D.nodes[x]['label'],
                    f'{(org_shares[x][property] * 100):.2f}',
                ) for i, x in enumerate(
                    sorted(
                        org_shares.keys(),
                        key=lambda x: org_shares[x][property],
                        reverse=True,
                    )[:NROWS])
            ]))

    with open('data/org-shares.tex', 'w') as fd:
        fd.write('\n'.join([
            '{} & {} & {} & {} \\\\'.format(
                D.nodes[x]['label'],
                *[('\\textbf{{{}}}' if y == max(org_shares[x].values()) else
                   '{}').format(f'{(y * 100):.2f}')
                  for y in org_shares[x].values()],
            ) for x in sorted(
                org_shares.keys(),
                key=lambda x: tuple(org_shares[x].values()),
                reverse=True,
            )[:NROWS]
        ]))

    with open('data/cls-shares.tex', 'w') as fd:
        fd.write('\n'.join([
            '({}) & {} & {} & {} \\\\'.format(
                ', '.join(clusters[x]['tfidf']),
                *[('\\textbf{{{}}}' if y == max(cls_shares[x].values()) else
                   '{}').format(f'{(y * 100):.2f}')
                  for y in cls_shares[x].values()],
            ) for x in sorted(
                cls_shares.keys(),
                key=lambda x: tuple(cls_shares[x].values()),
                reverse=True,
            )
        ]))

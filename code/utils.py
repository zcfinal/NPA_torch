import pandas as pd
from sklearn.cluster import estimate_bandwidth, MeanShift
import os
import pickle

def get_clusters(coords,args):
    """
    Estimate clusters for the given list of coordinates.
    """
    cluster = None
    if os.path.isfile(args.cluster_dir):
        with open(args.cluster_dir, 'rb') as handle:
            cluster = pickle.load(handle)
    else:
        # First, grossly reduce the spatial dataset by rounding up the coordinates to the 4th decimal
        # (i.e. 11 meters. See: https://en.wikipedia.org/wiki/Decimal_degrees)
        clusters = pd.DataFrame({
            'approx_latitudes': coords[:, 0].round(5),
            'approx_longitudes': coords[:, 1].round(5)
        })
        clusters = clusters.drop_duplicates(['approx_latitudes', 'approx_longitudes'])
        clusters = clusters.to_numpy()

        # Further reduce the number of clusters
        # (Note: the quantile parameter was tuned to find a significant and reasonable number of clusters)
        ms = MeanShift(bandwidth=args.bandwidth, bin_seeding=True,min_bin_freq=5)
        ms.fit(clusters)
        with open(args.cluster_dir, 'wb') as handle:
            pickle.dump(ms.cluster_centers_, handle, protocol=pickle.HIGHEST_PROTOCOL)
        cluster = ms.cluster_centers_
    print('the number of clusters:{}'.format(cluster.shape[0]))
    return cluster



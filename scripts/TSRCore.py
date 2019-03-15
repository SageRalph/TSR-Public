"""
This script contains an implementation of the TSR inference algorithm.
To use the infer function, you will first need a list of items with distances,
these can be calculated using the distancesSemantic function.
"""
import sklearn.preprocessing as preprocessing
import sklearn.metrics as metrics
import numpy as np


def distancesSemantic(items):
    """
    Adds to each item an ordered list of distances to all other items.
    The distance is 1 - the cosine similarity of their embeddings.
    """

    embeddings = [item['embedding'] for item in items]
    print("\nCALCULATING COSINE DISTANCES...")
    distances = metrics.pairwise.cosine_distances(embeddings, embeddings)

    for i in range(0, len(items)):

        # list of shape [distance, id]
        pairs = [(d, item['id']) for d, item in zip(distances[i], items)]

        del pairs[i]  # Ignore distance to self

        items[i]['distances'] = sorted(pairs, key=lambda x: x[0])

    return items


def infer(max_similar, max_related, query, items, allowed_target_ids, relation_type, mode):
    """
    Ranks targets by the distance from the input to the target passing
    through exactly one relation which is given as a distance of 0.
    Approx comparisons = min(max_similar, labelled items) * min(max_related, items) * average labels per item
    items should be dicts with the keys 'id' and 'distances', an ordered list of distances to other items
    if there is no distance entry from one item to another, it is assumed to be unreachable
    """

    L1 = max_similar
    L2 = max_related
    H = query
    O = []

    labelled_items = itemsWithKeys(items, [relation_type])
    labelled_ids = [item['id'] for item in labelled_items]

    # For each item Si of the L1 labelled items most similar to H
    S = [row for row in H['distances'] if row[1] in labelled_ids]
    if (L1):
        S = S[:L1]
    for D1, Si_id in S:
        Si = getNode(Si_id, labelled_items)
        if Si is None:
            continue

        # For each item Ri related to Si
        R = Si[relation_type]
        for Ri_id in R:
            Ri = getNode(Ri_id, items)
            if Ri is None:
                continue

            # Add to output the related node (if it is an allowed target)
            # The distance score is the distance from H to Si
            if(Ri_id in allowed_target_ids):
                match = {
                    'target_node': Ri,
                    'similar_node': Si,
                    'related_node': Ri,
                    'distance': D1
                }
                O.append(match)

            # For each item Ti of the L2 items most similar to Ri (that are allowed targets)
            T = [Ti for Ti in Ri['distances'] if Ti[1] in allowed_target_ids]
            if (L2):
                T = T[:L2]
            for D2, Ti_id in T:
                Ti = getNode(Ti_id, items)
                if Ti is None:
                    continue

                # Add to output the target node
                # The total distance is the sum of the distance from H to Si and Ri to Ti
                match = {
                    'target_node': Ti,
                    'similar_node': Si,
                    'related_node': Ri,
                    'distance': D1 + D2
                }
                O.append(match)

    return _scoreRoutes(O, mode)


def _scoreRoutes(collection, mode):
    """
    Converts a collection of identified routes into an order list of scored targets
    """
    outputs = []
    rangeFit = True  # Most algorithms need fitting to 0-1

    # Group routes by target node
    for value in collection:

        tID = value['target_node']['id']

        target = next(
            (item for item in outputs if item['target_id'] == tID), None)

        if not target:
            # Add target entry if this is it's first route
            outputs.append({
                'target_id': tID,
                'routes': [value]
            })
        else:
            # Otherwise add it to the routes list for the target
            target['routes'].append(value)

    for target in outputs:
        # Arrange routes shortest first
        target['routes'] = sorted(
            target['routes'], key=lambda k: k['distance'])
        # Find shortest distance
        target['distance'] = target['routes'][0]['distance']

    # Determine score for each target
    for target in outputs:

        if mode == 'a':
            # Sort by shortest distance (not range fitted)
            target['score'] = 1 - target['distance']/2
            rangeFit = False

        elif mode == 'a*':
            # Sort by shortest distance
            target['score'] = 1 - target['distance']/2

        elif mode == 'b':
            # Sort by number of routes to the target, followed by shortest distance
            target['score'] = len(target['routes']) - target['distance']/2

        elif mode == 'c':
            # Sort by most similarity * number of routes
            target['score'] = (1-target['distance']/2) * len(target['routes'])

        elif mode == 'd':
            # Sort by sum similarity over all routes
            target['score'] = sum(1 - r['distance'] for r in target['routes'])

        elif mode == 'e':
            # Sort by most similarity + second most similarity/2
            s = 1 - target['distance']/2
            if (len(target['routes']) > 1):
                s += (1 - target['routes'][1]['distance']/2)/2
            target['score'] = s

        elif mode == 'f':
            # Sort by sum over all routes of similarity/position
            s = 0
            for i in range(len(target['routes'])):
                r = target['routes'][i]
                j = i+1
                d = r['distance']/2
                s += (1 - d) / j
            target['score'] = s

        elif mode == 'g':
            # Sort by sum over all routes of similarity/position^2
            s = 0
            for i in range(len(target['routes'])):
                r = target['routes'][i]
                j = i+1
                d = r['distance']/2
                s += (1 - d) / (j*j)
            target['score'] = s

        elif mode == 'h':
            # Sort by sum over all routes of similarity/position^3
            s = 0
            for i in range(len(target['routes'])):
                r = target['routes'][i]
                j = i+1
                d = r['distance']/2
                s += (1 - d) / (j*j*j)
            target['score'] = s

        elif mode == 'i':
            # Sort by sum over all routes of 1-distance^2
            s = 0
            for i in range(len(target['routes'])):
                r = target['routes'][i]
                d = r['distance']/2
                s += (1 - d*d)
            target['score'] = s

        elif mode == 'j':
            # Sort by sum over all routes of 1-distance^3
            s = 0
            for i in range(len(target['routes'])):
                r = target['routes'][i]
                d = r['distance']/2
                s += (1 - d*d*d)
            target['score'] = s

        elif mode == 'k':
            # Sort by geometric series weighted sum similarity
            s = 0
            den = 2
            for r in target['routes']:
                s += (1 - r['distance']/2) / den
                den = den*2
            target['score'] = s

        elif mode == 'l':
            # Sort by telescopic weighted sum similarity
            s = 0
            for i in range(len(target['routes'])):
                r = target['routes'][i]
                n = i+1
                routeScore = 1 - r['distance']/2
                s += routeScore / (n*(n+1))
            s = s/2
            target['score'] = s

        elif mode == 'm':
            # Sort by sum over all routes of 1/(distance*position^3)
            s = 0
            for i in range(len(target['routes'])):
                r = target['routes'][i]
                j = i+1
                d = r['distance']/2
                s += 1 / (d*j*j*j)
            target['score'] = s

        elif mode == 'n':
            # Sort by sum over all routes of 1/distance^2
            s = 0
            for i in range(len(target['routes'])):
                r = target['routes'][i]
                j = i+1
                d = r['distance']/2
                s += 1 / (d*d)
            target['score'] = s

        elif mode == 'o':
            # Sort by sum over all routes of 1/(distance*position)
            s = 0
            for i in range(len(target['routes'])):
                r = target['routes'][i]
                j = i+1
                d = r['distance']/2
                s += 1 / (d*j)
            target['score'] = s

        elif mode == 'p':
            # Sort by sum over all routes of 1/(distance*position^2)
            s = 0
            for i in range(len(target['routes'])):
                r = target['routes'][i]
                j = i+1
                d = r['distance']/2
                s += 1 / (d*j*j)
            target['score'] = s

        elif mode == 'q':
            # Sort by sum over all routes of 1/distance
            target['score'] = sum(
                1/(r['distance']/2) for r in target['routes'])

    if(rangeFit):
        # Fit all scores to the range 0-1
        scores = np.array([target['score'] for target in outputs])
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        scaled = min_max_scaler.fit_transform(scores[:, np.newaxis])

        for i in range(0, len(outputs)):
            outputs[i]['score'] = scaled[i][0]

    # Return results in descending order of score
    return sorted(outputs, key=lambda k: -k['score'])


def getNode(id, items):
    """
    Gets dictionary with id from a list of dictionaries
    """
    return next((item for item in items if item["id"] == id), None)


def itemsWithKeys(items, labels):
    """
    Returns a list of items with non falsy values for all labels
    """
    return [item for item in items if all(k in item and item[k] for k in labels)]

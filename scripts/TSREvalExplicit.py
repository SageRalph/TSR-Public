"""
This script uses TSRCore to perform leave-one-out cross validation of 
the TSR inference algorithm. Most options can be specified by commandline.
Scoring uses R-Precision and error values.
"""
import ntpath
import numpy as np
import copy
import argparse
import sklearn.metrics as metrics
import TSRCore as core
import util


def main():

    # Get inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", "--in",
                        help="Path to JSON file containing labelled items with embeddings")
    parser.add_argument("--out", "-o", "--output",
                        help="Path for CSV file to output results")
    parser.add_argument("--pos", "-p", "--positive",
                        help="Name of positive relation label")
    parser.add_argument("--neg", "-n", "--negative",
                        help="Name of negative relation label")
    parser.add_argument("--mode", "-m",
                        help="Scoring algorithm (a to q)")

    args = parser.parse_args()

    inPath = args.input or input(
        "\nENTER PATH OF INPUT FILE:\n")

    relation_pos = args.pos or input(
        "\nENTER NAME OF POSITIVE RELATION LABEL:\n")

    relation_neg = args.neg or input(
        "\nENTER NAME OF NEGATIVE RELATION LABEL:\n")

    mode = args.mode or input(
        "\nSELECT SCORING ALGORITHM:\n")

    outPath = args.out

    items = util.readJSONFile(inPath)

    # Pre-calculate cosine distance for all items
    items = core.distancesSemantic(items)

    # We can only evaluate labelled items
    labelled = core.itemsWithKeys(items, [relation_pos, relation_neg])
    print(f'\nFOUND {len(labelled)} LABELLED ITEMS\n')
    if not len(labelled):
        return

    dsname = ntpath.basename(inPath)
    r = evaluateItems(labelled, items, relation_pos,
                      relation_neg, mode, dsname)

    print('\n' + r['text'])

    if outPath:
        util.writeCSV(outPath, [r], ['text', 'P@R', 'R@R'])


def evaluateItems(labelled, items, relation_pos, relation_neg, mode, dsname):
    """
    Determine the rank score of each label for each labelled item
    We can evaluate performance by the mean rank for the labels
    """
    L1 = 5
    L2 = 10

    all_scores = [] # Predicted scores
    all_GT = [] # Ground Truth labels (0 or 1)
    all_PR = [] # Predicted labels (0 or 1)

    for selected in range(0, len(labelled)):

        query = labelled[selected]

        # Only rank items which the query has a known label for
        tests_pos = query[relation_pos]
        tests_neg = query[relation_neg]
        allowed_target_ids = tests_pos + tests_neg

        # Remove the query from the dataset
        safe_items = items.copy()
        safe_items.remove(query)

        # Strip all labels from the query
        safe_query = copy.deepcopy(query)
        safe_query[relation_pos] = []

        # Rank
        ranked = core.infer(
            max_similar=L1,
            max_related=L2,
            query=safe_query,
            items=safe_items,
            allowed_target_ids=allowed_target_ids,
            relation_type=relation_pos,
            mode=mode
        )

        threshold = len(tests_pos)  # Rank threshold for R-Precision
        worstRank = len(allowed_target_ids) - 1 # Default if item not in results

        ranked_ids = [item['target_id'] for item in ranked]

        # Check ranks of positive labels
        for id in tests_pos:

            # Add to Ground Truth
            all_GT.append(1)

            # Record label based on rank threshold
            rank = ranked_ids.index(id) if id in ranked_ids else worstRank
            all_PR.append(1 if rank < threshold else 0)

            # Record score
            score = ranked[rank]['score'] if id in ranked_ids else 0
            all_scores.append(score)

        # Check ranks of negative labels
        for id in tests_neg:

            # Add to Ground Truth
            all_GT.append(0)

            # Record label based on rank threshold
            rank = ranked_ids.index(id) if id in ranked_ids else worstRank
            all_PR.append(1 if rank < threshold else 0)

            # Record score
            score = ranked[rank]['score'] if id in ranked_ids else 0
            all_scores.append(score)

        print(f'EVALUATED QUERY: {str(safe_query["id"]).ljust(5)}')

    np.set_printoptions(precision=4)

    r = {
        'text': '',
        'dataset': dsname,
        'positive_label_name': relation_pos,
        'negative_label_name': relation_neg,
        'labelled_items_count': len(labelled),
        'positive_label_count': all_GT.count(1),
        'negative_label_count': all_GT.count(0),
        'scoring_mode': mode,
        'L1': L1,
        'L2': L2
    }

    r['text'] += f'\
FOR {r["labelled_items_count"]} ITEMS \
WITH {r["positive_label_count"]} POSITIVE "{r["positive_label_name"]}" LABELS \
AND {r["negative_label_count"]} NEGATIVE "{r["negative_label_name"]}" LABELS:'

    r['text'] += f'\nSCORING MODE: {r["scoring_mode"]} L1={r["L1"]} L2={r["L2"]}'

    # Chosen threshold
    r['text'] += f'\n\nEVALUATED @R (Threshold = len(pos) per item):'

    r['P@R'], r['R@R'], r['F1@R'], S = metrics.precision_recall_fscore_support(
        all_GT, all_PR, average='binary')
    r['text'] += f"\n\
    PRECISION@R: {r['P@R']:1.4f}\n\
    RECALL@R   : {r['R@R']:1.4f}\n\
    F1@R       : {r['F1@R']:1.4f}"

    r['TN@R'], r['FP@R'], r['FN@R'], r['TP@R'] = metrics.confusion_matrix(
        all_GT, all_PR).ravel()
    r['text'] += f"\n\
    CONFUSION MATRIX@R:\n\
        TP:{r['TP@R']} FP:{r['FP@R']}\n\
        FN:{r['FN@R']} TN:{r['TN@R']}"

    # No chosen threshold
    r['text'] += f'\n\nEVALUATED BY SCORE:'

    r['RMS_error'] = metrics.mean_squared_log_error(all_GT, all_scores)
    r['text'] += f'\n\
    RMS ERROR: {r["RMS_error"]:1.4f}'

    r['median_abs_error'] = metrics.median_absolute_error(all_GT, all_scores)
    r['text'] += f'\n\
    MEDIAN ABSOLUTE ERROR: {r["median_abs_error"]:1.4f}'

    return r


if __name__ == '__main__':
    main()

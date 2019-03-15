"""
This script uses TSRCore to perform implicit feedback 1-in-100 evaluation of
the TSR inference algorithm. Most options can be specified by commandline.
1-in-100 evaluation groups one known positive with 100 random items and scores
based on the algorithm's ability to rank the true positive highly.
This script can be time and resource intensive if the repeat count is high.
This script is optimised for multi-core CPUs.
"""
import ntpath
import numpy as np
import copy
import argparse
import TSRCore as core
import util
import multiprocessing
from functools import partial
import traceback


def main():

    # Get inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", "--in",
                        help="Path to JSON file containing labelled items with embeddings")
    parser.add_argument("--out", "-o", "--output",
                        help="Path for CSV file to output results")
    parser.add_argument("--rep", "-r", "--repeat",
                        help="Number of different random pools to rank each positive label against")
    parser.add_argument("--pos", "-p", "--positive",
                        help="Name of positive relation label")
    parser.add_argument("--mode", "-m",
                        help="Scoring algorithm (a to q)")

    args = parser.parse_args()

    inPath = args.input or input(
        "\nENTER PATH OF INPUT FILE:\n")

    attempts = int(args.rep or input(
        "\nENTER NUMBER OF RUNS/RANDOM POOLS PER POSITIVE LABEL:\n"))

    relation_pos = args.pos or input(
        "\nENTER NAME OF POSITIVE RELATION LABEL:\n")

    mode = args.mode or input(
        "\nSELECT SCORING ALGORITHM:\n")

    outPath = args.out

    items = util.readJSONFile(inPath)

    # Pre-calculate cosine distance for all items
    items = core.distancesSemantic(items)

    # We can only evaluate labelled items
    labelled = core.itemsWithKeys(items, [relation_pos])
    print(f'\nFOUND {len(labelled)} LABELLED ITEMS')
    if not len(labelled):
        return

    dsname = ntpath.basename(inPath)
    r = evaluateItems(labelled, items, relation_pos, mode, dsname, attempts)

    print('\n' + r['text'])

    if outPath:
        util.writeCSV(outPath, [r], ['text', 'positive_label_ranks'])


def evaluateItems(labelled, items, relation_pos, mode, dsname, attempts,
                  poolsize=101):
    """
    Determine the rank score of each label for each labelled item
    We can evaluate performance by the mean rank for the labels
    """
    L1 = 5
    L2 = 10

    # Genrate common framework test scenarios
    # 100 randomly chosen unknowns with 1 known positive mixed in
    # multiple attempts are made for each scenario
    cases = []
    for labelled_item in labelled:
        for pos in labelled_item[relation_pos]:
            for i in range(1, attempts+1):
                # Take 100 random items that do not have a known positive relation
                target_ids = [
                    t['id'] for t in items if t['id'] not in labelled_item[relation_pos]
                ]
                np.random.shuffle(target_ids)
                target_ids = target_ids[:poolsize-1]

                # Add one positive example and shuffle
                target_ids.append(pos)
                np.random.shuffle(target_ids)

                cases.append({
                    "query": labelled_item,
                    "pos_id": pos,
                    "target_ids": target_ids,
                    "attempt": i
                })

    # Run test cases in parallel
    runTest = partial(doCase,
                      items=items,
                      relation_pos=relation_pos,
                      mode=mode,
                      L1=L1,
                      L2=L2,
                      poolsize=poolsize)
    print('\nSPAWNING WORKER PROCESSES...')
    with multiprocessing.Pool() as pool:
        print('\nPROCESSING TEST CASES...\n')
        results = pool.map(runTest, cases)
    print('\nALL TEST CASES COMPLETE\n')

    # Unpack results
    all_pos_ranks = [rank for id, rank in results]

    # Group results per target
    per_pos_ranks = {}
    for id, rank in results:
        if id in per_pos_ranks:
            per_pos_ranks[id].append(rank)
        else:
            per_pos_ranks[id] = [rank]

    r = {
        'text': '',
        'dataset': dsname,
        'positive_label_name': relation_pos,
        'labelled_items_count': len(labelled),
        'positive_label_count': int(len(all_pos_ranks)/attempts),
        'evaluation_repeat_count': attempts,
        'scoring_mode': mode,
        'L1': L1,
        'L2': L2
    }

    r['positive_label_ranks'] = all_pos_ranks
    r['total_evaluations_count'] = len(all_pos_ranks)
    r['hits@10'] = len([1 for i in all_pos_ranks if i < 10])
    r['HR@10'] = r['hits@10'] / r['total_evaluations_count']
    r['hits@5'] = len([1 for i in all_pos_ranks if i < 5])
    r['HR@5'] = r['hits@5'] / r['total_evaluations_count']
    r['hits@1'] = len([1 for i in all_pos_ranks if i < 1])
    r['HR@1'] = r['hits@1'] / r['total_evaluations_count']
    r['median_label_positive_rank'] = np.median(all_pos_ranks)
    r['mean_label_positive_rank'] = util.mean(all_pos_ranks)

    r['text'] += f'\
FOR {r["labelled_items_count"]} ITEMS \
WITH {r["positive_label_count"]} POSITIVE "{r["positive_label_name"]}" LABELS \
EACH RANKED OUT OF {poolsize-1} RANDOM ITEMS {attempts} TIMES'

    r['text'] += f'\nSCORING MODE: {r["scoring_mode"]} L1={r["L1"]} L2={r["L2"]}'

    r['text'] += f'\nMEDIAN POSITIVE LABEL RANK: {r["median_label_positive_rank"]}'

    r['text'] += f'\nMEAN POSITIVE LABEL RANK: {r["mean_label_positive_rank"]:1.4f}'

    r['text'] += f'\nHIT RATE @10: {r["HR@10"]:1.4f}'

    r['text'] += f'\nHIT RATE @5:  {r["HR@5"]:1.4f}'

    r['text'] += f'\nHIT RATE @1:  {r["HR@1"]:1.4f}'

    return r


def doCase(case, items, relation_pos, mode, L1, L2, poolsize):
    try:

        pos_id = case["pos_id"]
        query = case["query"]

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
            allowed_target_ids=case["target_ids"],
            relation_type=relation_pos,
            mode=mode
        )

        # Determine the ranking of the known positive
        ranked_ids = [item["target_id"] for item in ranked]
        pos_rank = ranked_ids.index(
            pos_id) if pos_id in ranked_ids else poolsize

        print(f'\
QUERY: {str(safe_query["id"]).ljust(5)} \
TARGET: {str(pos_id).ljust(5)} \
ATTEMPT: {str(case["attempt"]).ljust(5)} \
POSITIVE LABEL RANK: {pos_rank}')

        return (pos_id, pos_rank)

    except:
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()

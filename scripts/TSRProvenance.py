"""
This script uses TSRCore to rank targets for a chosen query item and outputs 
a detailed provenance file showing the scores, routes, and descriptions for all targets.
"""
import copy
import argparse
import TSRCore as core
import util


def main():

    # Get inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", "--in",
                        help="Path to JSON file containing labelled items with embeddings")
    parser.add_argument("--out", "-o", "--output",
                        help="Path for directory for output results file")
    parser.add_argument("--pos", "-p", "--positive",
                        help="Name of positive relation label")
    parser.add_argument("--mode", "-m",
                        help="Scoring algorithm (a to q)")
    parser.add_argument("--query", "-q", type=int,
                        help="Index of the query. Items will be listed on start if not set")

    args = parser.parse_args()

    inPath = args.input or input(
        "\nENTER PATH OF INPUT FILE:\n")

    relation_pos = args.pos or input(
        "\nENTER NAME OF POSITIVE RELATION LABEL:\n")

    mode = args.mode or input(
        "\nSELECT SCORING ALGORITHM (input the letter):")

    outPath = args.out or '.'
    queryIndex = args.query

    items = util.readJSONFile(inPath)

    # Pre-calculate cosine distance for all items
    items = core.distancesSemantic(items)

    query = getQuery(items, queryIndex)

    # All items except the query
    target_ids = [item['id'] for item in items if item is not query]

    # Remove the query from the dataset
    safe_items = items.copy()
    safe_items.remove(query)

    # Strip all labels from the query
    safe_query = copy.deepcopy(query)
    safe_query[relation_pos] = []

    # Rank
    ranked = core.infer(
        max_similar=5,
        max_related=10,
        query=safe_query,
        items=safe_items,
        allowed_target_ids=target_ids,
        relation_type=relation_pos,
        mode=mode
    )

    outFile = f"{outPath}/{query['name'].replace('/',' ')}.{relation_pos}.TSR-{mode}.txt"
    outputScores(ranked, query, outFile)


def getQuery(items, index):
    """
    Prompts user to select one of items
    items must be a list of dicts, all with the property 'name'
    """
    valid = False
    while(not valid):
        try:
            if(not index):
                # List known items
                print("\nKNOWN ITEMS:")
                for i in range(0, len(items)):
                    print(f"{i}  {items[i]['name']}")

                # Prompt for query
                index = input("\nENTER INDEX TO SELECT:\n")

            # Test if an index entered
            selected = int(index)
            item = items[selected]

            # input was an index, return the item's embedding
            print(f"\nSELECTED: {item['name']}")
            return item

        except ValueError:
            print("\nINVALID SELECTION\n")
            index = None


def outputScores(items, query, out_file):
    """
    Prints and saves to file the names and scores of items
    """

    text = f"POTENTIAL RELATIONS FOR:\n\
QUERY NAME: {query['name']}\n\
DESCRIPTION: {query['description']}"

    for result in items:
        t_node = result['routes'][0]['target_node']

        text += f"\n\n\
SCORE: {result['score']:1.4f}\n\
TARGET NAME: {t_node['name']}\n\
TARGET ID: {t_node['id']}\n\
DESCRIPTION: {t_node['description']}\n\
ROUTES:"

        for route in result['routes']:
            s_name = route['similar_node']['name']
            r_name = route['related_node']['name']
            similarity = 1-route['distance']/2
            text += f"\n{similarity:1.4f}    SIMILAR: {s_name.ljust(32)}    RELATED: {r_name}"

    if out_file:
        print('\nSAVING TO FILE: '+out_file)
        with open(out_file, "w") as text_file:
            text_file.write(text)
    else:
        print(text)


if __name__ == '__main__':
    main()

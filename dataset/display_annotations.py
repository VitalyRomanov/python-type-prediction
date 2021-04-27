import spacy 
import sys
import json
from spacy.gold import biluo_tags_from_offsets
from spacy.tokenizer import Tokenizer
import re

def custom_tokenizer(nlp):
    prefix_re = re.compile(r'''^[\[\(\{"':\.!@~,=+-/\*]''')
    suffix_re = re.compile(r'''[\]\)\}"':\.\!~,=+-/\*]$''')
    infix_re = re.compile(r'''[\[\]\(\)\{\},=+-/@\*\"\.!:~]''')
    return Tokenizer(nlp.vocab,
                                prefix_search=prefix_re.search,
                                suffix_search=suffix_re.search,
                                infix_finditer=infix_re.finditer,
                                )

def inject_tokenizer(nlp):
    nlp.tokenizer = custom_tokenizer(nlp)
    return nlp

nlp = inject_tokenizer(spacy.blank("en"))

annotations_path = sys.argv[1]

with open(annotations_path) as annotations:
    for line in annotations:
        entry = json.loads(line.strip())
        # json schema
        # {
        #     text: stores code
        #     ents: stores type annotations in spacy NER format
        #     cats: function return type (can have several if there are nested function definitions)
        #     docstrings: stores docstrings, for main and nested functions
        #     replacements: used for another project
        # }
        doc = nlp(entry['text']) # store
        tags = biluo_tags_from_offsets(doc, entry['ents'])
        for t, tag in zip(doc, tags):
            print(t.text, tag, sep="\t\t")
        print()
        
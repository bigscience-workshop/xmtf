from functools import partial
import json
import multiprocessing
import os
import random

from datasets import load_dataset
from datasets import get_dataset_config_names
# pip install -q iso-639
from iso639 import languages
# pip install git+https://github.com/Muennighoff/promptsource.git@xp3x
from promptsource.templates import DatasetTemplates

# Set to False to use multilingual prompts e.g. 'id' for xcopa/id instead of 'en'
USE_ENGLISH_PROMPTS = True

MAX_EXAMPLES_PER_DATASET_PROMPT = 100_000

STORY_CLOZE_DIR = "./story_cloze_data"
XSTORY_CLOZE_DIR = "./xstory_cloze_data"

# Some datasets have test sets with hidden labels which will still compile but only to noise
# e.g. piqa test labels are all [-1] which still works on list indices resulting in 
# noise samples where the label is always the same  
SKIP_PROMPTS = {
    "common_gen": {"test": ["all"]},
    "piqa": {"test": ["all"]},
    "qasc": {"test": ["all"]},
    "imdb": {"unsupervised": ["all"]},
    "glue/qqp": {"test": ["all"]},
    "super_glue/record": {"test": ["all"]},
    "qasc": {"test": ["all"]},
    'kilt_tasks/hotpotqa': {"test": ["all"]},
    "cosmos_qa": {"test": [
        "description_context_question_answer_text", 
        "description_context_question_text",
        "description_context_question_answer_id",
        "context_answer_to_question",
        "context_description_question_answer_text",
        "context_description_question_answer_id",
        "context_question_description_answer_id",
        "context_description_question_text",
        "context_question_description_answer_text",
        "only_question_answer",
        "no_prompt_id",
        "context_question_description_text",
        "no_prompt_text",
        ]},
    "clue/tnews": {"test": ["all"]},
    "clue/csl": {"test": ["all"]},
    "clue/cmrc2018": {"test": ["generate_question", "in_an_exam", "answer_in_the_passage", "answer_following_question", "xp3longcontinue"]},
    "clue/drcd": {"test": ["generate_question", "in_an_exam", "answer_in_the_passage", "answer_following_question", "xp3longcontinue"]},
    "hellaswag": {"test": ["complete_first_then", "Topic of the context", "Open-ended completion", "Randomized prompts template", "Appropriate continuation - Yes or No", "Predict ending with hint", "Open-ended start", "Reversed appropriate continuation - Yes or No", "how_ends", "if_begins_how_continues"]},
}

DS_TO_ENG_PROMPT = {
    "xcopa": "en",
    "Muennighoff/xstory_cloze": "en",
    "Muennighoff/xwinograd": "en",
    'GEM/wiki_lingua': 'en_en', # Contains correct language names
    'facebook/flores': 'x_x', # Contains correct language names    
    "allenai/wmt22_african": "x_x",
    "Helsinki-NLP/tatoeba_mt": "x_x",
    "Muennighoff/multi_eurlex": "x_x",
    'xnli': 'en',
    "paws-x": "en",
    "mlqa": "mlqa.en.en",
    "xquad": "xquad.en",
    "khalidalt/tydiqa-primary": "english",
    "khalidalt/tydiqa-goldp": "english",
    "pasinit/xlwic": "en",
    "GEM/xlsum": "english",
    "GEM/BiSECT": "en",
}

TRAIN_DATASETS_EXT = [
    # Multilingual; Iterate over all configs
    'Muennighoff/xwinograd',
    'Muennighoff/xstory_cloze',
    'xcopa',
    'xnli',
    'paws-x',
    'mlqa',
    'xquad',
    'khalidalt/tydiqa-primary',
    'khalidalt/tydiqa-goldp',
    'pasinit/xlwic',
    'GEM/xlsum',
    'Helsinki-NLP/tatoeba_mt',
    'GEM/BiSECT',
    'allenai/wmt22_african',
    "GEM/wiki_lingua",
]

#TRAIN_DATASETS_EXT = ['Helsinki-NLP/tatoeba_mt']

# Datasets for which to use specific configs; Else use all configs
DATASET_TO_CONFIGS = {
    # Ignore the translation configs
    "xcopa": ["et", "ht", "it", "id", "qu", "sw", "zh", "ta", "th", "tr", "vi"],
}

DATASET_TO_SKIP_CONFIGS = {
    "GEM/wiki_lingua": ["multilingual", "crosslingual"],
    "xnli": ["all_languages"],
    "mutli_eurlex": ["all_languages"],
}

TRAIN_DATASETS_EN = [
    # English-only
    ('glue','mrpc'), 
    ('glue','qqp'),
    ('paws','labeled_final'),
    ('ai2_arc','ARC-Challenge'),
    ('ai2_arc','ARC-Easy'),
    ('kilt_tasks','hotpotqa'),
    ('trivia_qa','unfiltered'),
    ('web_questions',None),
    ('wiki_qa',None),
    ('adversarial_qa','dbidaf'),
    ('adversarial_qa','dbert'),
    ('adversarial_qa','droberta'),
    ('duorc','SelfRC'),
    ('duorc','ParaphraseRC'),
    ('ropes',None),
    ('squad_v2',None),
    ('super_glue','record'),
    ('quoref',None),
    ('cos_e','v1.11'),
    ('cosmos_qa',None),
    ('dream',None),
    ('openbookqa','main'),
    ('qasc',None),
    ('quail',None),
    ('quarel',None),
    ('quartz',None),
    ('race','high'),
    ('race','middle'),
    ('sciq',None),
    ('social_i_qa',None),
    ('super_glue','boolq'),
    ('super_glue','multirc'),
    ('wiki_hop','original'),
    ('wiqa',None),
    ('piqa',None),
    ('amazon_polarity',None),
    ('app_reviews',None),
    ('imdb',None),
    ('rotten_tomatoes',None),
    ('yelp_review_full',None),
    ('common_gen',None),
    ('wiki_bio',None),
    ('cnn_dailymail','3.0.0'),
    ('gigaword',None),
    ('multi_news',None),
    ('samsum',None),
    ('xsum',None),
    ('ag_news',None),
    ('dbpedia_14',None),
    ('trec',None),

    ('super_glue', 'wic'),
    ('hellaswag', None),
    ('super_glue', 'copa'),  
    ('super_glue','wsc.fixed'),
    ('winogrande','winogrande_xl'),
    ("anli", None),
    ("super_glue", "rte"),
    ("super_glue", "cb"),
    # ('story_cloze', '2016'), # Not public 
]

TRAIN_DATASETS = [
    ('Muennighoff/mbpp', 'sanitized'),
    ("great_code", None),
    ("neural_code_search", "evaluation_dataset"),
    ("codeparrot/codecomplex", "codeparrot--codecomplex"),
    ("codeparrot/github-jupyter-text-code-pairs", None),
    ("codeparrot/apps", "all"),
    ("codeparrot/xlcost-text-to-code", "Python-program-level"),
    ("codeparrot/xlcost-text-to-code", "C-program-level"),
    ("codeparrot/xlcost-text-to-code", "C++-program-level"),
    ("codeparrot/xlcost-text-to-code", "Csharp-program-level"),
    ("codeparrot/xlcost-text-to-code", "Java-program-level"),
    ("codeparrot/xlcost-text-to-code", "Javascript-program-level"),
    ("codeparrot/xlcost-text-to-code", "PHP-program-level"),
    ("teven/code_contests", None),
    ("teven/code_docstring_corpus", "top_level"),
    ("Fraser/python-state-changes", None),
    ('clue', 'c3'),
    ('clue', 'cmrc2018'),
    ('clue', 'csl'),
    ('clue', 'drcd'),
    ('clue', 'tnews'),
] + TRAIN_DATASETS_EN



#TRAIN_DATASETS = []
#"""
for ds in TRAIN_DATASETS_EXT:
    if ds in DATASET_TO_CONFIGS:
        TRAIN_DATASETS.extend([(ds, conf) for conf in DATASET_TO_CONFIGS[ds]])
    else:
        TRAIN_DATASETS.extend([(ds, conf) for conf in get_dataset_config_names(ds) if not conf in DATASET_TO_SKIP_CONFIGS.get(ds, [])])
        if ('Helsinki-NLP/tatoeba_mt' in ds) or ('allenai/wmt22_african' in ds):
            # Mark for inversion
            TRAIN_DATASETS.extend([(ds, conf + "-inverted") for conf in get_dataset_config_names(ds)])

#TRAIN_DATASETS = [
#    #(ds, conf) for (ds, conf) in TRAIN_DATASETS if conf is not None and "-inverted" in conf
#]
#"""
print("TRAIN_DATASETS", TRAIN_DATASETS)

# https://github.com/facebookresearch/flores/blob/main/flores200/README.md
FLORES_LANGS = {'Acehnese (Arabic script)': 'ace_Arab', 'Acehnese (Latin script)': 'ace_Latn', 'Mesopotamian Arabic': 'acm_Arab', 'Ta’izzi-Adeni Arabic': 'acq_Arab', 'Tunisian Arabic': 'aeb_Arab', 'Afrikaans': 'afr_Latn', 'South Levantine Arabic': 'ajp_Arab', 'Akan': 'aka_Latn', 'Amharic': 'amh_Ethi', 'North Levantine Arabic': 'apc_Arab', 'Modern Standard Arabic': 'arb_Arab', 'Modern Standard Arabic (Romanized)': 'arb_Latn', 'Najdi Arabic': 'ars_Arab', 'Moroccan Arabic': 'ary_Arab', 'Egyptian Arabic': 'arz_Arab', 'Assamese': 'asm_Beng', 'Asturian': 'ast_Latn', 'Awadhi': 'awa_Deva', 'Central Aymara': 'ayr_Latn', 'South Azerbaijani': 'azb_Arab', 'North Azerbaijani': 'azj_Latn', 'Bashkir': 'bak_Cyrl', 'Bambara': 'bam_Latn', 'Balinese': 'ban_Latn', 'Belarusian': 'bel_Cyrl', 'Bemba': 'bem_Latn', 'Bengali': 'ben_Beng', 'Bhojpuri': 'bho_Deva', 'Banjar (Arabic script)': 'bjn_Arab', 'Banjar (Latin script)': 'bjn_Latn', 'Standard Tibetan': 'bod_Tibt', 'Bosnian': 'bos_Latn', 'Buginese': 'bug_Latn', 'Bulgarian': 'bul_Cyrl', 'Catalan': 'cat_Latn', 'Cebuano': 'ceb_Latn', 'Czech': 'ces_Latn', 'Chokwe': 'cjk_Latn', 'Central Kurdish': 'ckb_Arab', 'Crimean Tatar': 'crh_Latn', 'Welsh': 'cym_Latn', 'Danish': 'dan_Latn', 'German': 'deu_Latn', 'Southwestern Dinka': 'dik_Latn', 'Dyula': 'dyu_Latn', 'Dzongkha': 'dzo_Tibt', 'Greek': 'ell_Grek', 'English': 'eng_Latn', 'Esperanto': 'epo_Latn', 'Estonian': 'est_Latn', 'Basque': 'eus_Latn', 'Ewe': 'ewe_Latn', 'Faroese': 'fao_Latn', 'Fijian': 'fij_Latn', 'Finnish': 'fin_Latn', 'Fon': 'fon_Latn', 'French': 'fra_Latn', 'Friulian': 'fur_Latn', 'Nigerian Fulfulde': 'fuv_Latn', 'Scottish Gaelic': 'gla_Latn', 'Irish': 'gle_Latn', 'Galician': 'glg_Latn', 'Guarani': 'grn_Latn', 'Gujarati': 'guj_Gujr', 'Haitian Creole': 'hat_Latn', 'Hausa': 'hau_Latn', 'Hebrew': 'heb_Hebr', 'Hindi': 'hin_Deva', 'Chhattisgarhi': 'hne_Deva', 'Croatian': 'hrv_Latn', 'Hungarian': 'hun_Latn', 'Armenian': 'hye_Armn', 'Igbo': 'ibo_Latn', 'Ilocano': 'ilo_Latn', 'Indonesian': 'ind_Latn', 'Icelandic': 'isl_Latn', 'Italian': 'ita_Latn', 'Javanese': 'jav_Latn', 'Japanese': 'jpn_Jpan', 'Kabyle': 'kab_Latn', 'Jingpho': 'kac_Latn', 'Kamba': 'kam_Latn', 'Kannada': 'kan_Knda', 'Kashmiri (Arabic script)': 'kas_Arab', 'Kashmiri (Devanagari script)': 'kas_Deva', 'Georgian': 'kat_Geor', 'Central Kanuri (Arabic script)': 'knc_Arab', 'Central Kanuri (Latin script)': 'knc_Latn', 'Kazakh': 'kaz_Cyrl', 'Kabiyè': 'kbp_Latn', 'Kabuverdianu': 'kea_Latn', 'Khmer': 'khm_Khmr', 'Kikuyu': 'kik_Latn', 'Kinyarwanda': 'kin_Latn', 'Kyrgyz': 'kir_Cyrl', 'Kimbundu': 'kmb_Latn', 'Northern Kurdish': 'kmr_Latn', 'Kikongo': 'kon_Latn', 'Korean': 'kor_Hang', 'Lao': 'lao_Laoo', 'Ligurian': 'lij_Latn', 'Limburgish': 'lim_Latn', 'Lingala': 'lin_Latn', 'Lithuanian': 'lit_Latn', 'Lombard': 'lmo_Latn', 'Latgalian': 'ltg_Latn', 'Luxembourgish': 'ltz_Latn', 'Luba-Kasai': 'lua_Latn', 'Ganda': 'lug_Latn', 'Luo': 'luo_Latn', 'Mizo': 'lus_Latn', 'Standard Latvian': 'lvs_Latn', 'Magahi': 'mag_Deva', 'Maithili': 'mai_Deva', 'Malayalam': 'mal_Mlym', 'Marathi': 'mar_Deva', 'Minangkabau (Arabic script)': 'min_Arab', 'Minangkabau (Latin script)': 'min_Latn', 'Macedonian': 'mkd_Cyrl', 'Plateau Malagasy': 'plt_Latn', 'Maltese': 'mlt_Latn', 'Meitei (Bengali script)': 'mni_Beng', 'Halh Mongolian': 'khk_Cyrl', 'Mossi': 'mos_Latn', 'Maori': 'mri_Latn', 'Burmese': 'mya_Mymr', 'Dutch': 'nld_Latn', 'Norwegian Nynorsk': 'nno_Latn', 'Norwegian Bokmål': 'nob_Latn', 'Nepali': 'npi_Deva', 'Northern Sotho': 'nso_Latn', 'Nuer': 'nus_Latn', 'Nyanja': 'nya_Latn', 'Occitan': 'oci_Latn', 'West Central Oromo': 'gaz_Latn', 'Odia': 'ory_Orya', 'Pangasinan': 'pag_Latn', 'Eastern Panjabi': 'pan_Guru', 'Papiamento': 'pap_Latn', 'Western Persian': 'pes_Arab', 'Polish': 'pol_Latn', 'Portuguese': 'por_Latn', 'Dari': 'prs_Arab', 'Southern Pashto': 'pbt_Arab', 'Ayacucho Quechua': 'quy_Latn', 'Romanian': 'ron_Latn', 'Rundi': 'run_Latn', 'Russian': 'rus_Cyrl', 'Sango': 'sag_Latn', 'Sanskrit': 'san_Deva', 'Santali': 'sat_Olck', 'Sicilian': 'scn_Latn', 'Shan': 'shn_Mymr', 'Sinhala': 'sin_Sinh', 'Slovak': 'slk_Latn', 'Slovenian': 'slv_Latn', 'Samoan': 'smo_Latn', 'Shona': 'sna_Latn', 'Sindhi': 'snd_Arab', 'Somali': 'som_Latn', 'Southern Sotho': 'sot_Latn', 'Spanish': 'spa_Latn', 'Tosk Albanian': 'als_Latn', 'Sardinian': 'srd_Latn', 'Serbian': 'srp_Cyrl', 'Swati': 'ssw_Latn', 'Sundanese': 'sun_Latn', 'Swedish': 'swe_Latn', 'Swahili': 'swh_Latn', 'Silesian': 'szl_Latn', 'Tamil': 'tam_Taml', 'Tatar': 'tat_Cyrl', 'Telugu': 'tel_Telu', 'Tajik': 'tgk_Cyrl', 'Tagalog': 'tgl_Latn', 'Thai': 'tha_Thai', 'Tigrinya': 'tir_Ethi', 'Tamasheq (Latin script)': 'taq_Latn', 'Tamasheq (Tifinagh script)': 'taq_Tfng', 'Tok Pisin': 'tpi_Latn', 'Tswana': 'tsn_Latn', 'Tsonga': 'tso_Latn', 'Turkmen': 'tuk_Latn', 'Tumbuka': 'tum_Latn', 'Turkish': 'tur_Latn', 'Twi': 'twi_Latn', 'Central Atlas Tamazight': 'tzm_Tfng', 'Uyghur': 'uig_Arab', 'Ukrainian': 'ukr_Cyrl', 'Umbundu': 'umb_Latn', 'Urdu': 'urd_Arab', 'Northern Uzbek': 'uzn_Latn', 'Venetian': 'vec_Latn', 'Vietnamese': 'vie_Latn', 'Waray': 'war_Latn', 'Wolof': 'wol_Latn', 'Xhosa': 'xho_Latn', 'Eastern Yiddish': 'ydd_Hebr', 'Yoruba': 'yor_Latn', 'Yue Chinese': 'yue_Hant', 'Chinese (Simplified)': 'zho_Hans', 'Chinese (Traditional)': 'zho_Hant', 'Standard Malay': 'zsm_Latn', 'Zulu': 'zul_Latn'}
FLORES_LANGS_INV = {v: k for k, v in FLORES_LANGS.items()}

FLORES_NEW_TO_OLD = {'afr_Latn': 'afr', 'amh_Ethi': 'amh', 'arb_Arab': 'ara', 'asm_Beng': 'asm', 'ast_Latn': 'ast', 'azj_Latn': 'azj', 'bel_Cyrl': 'bel', 'ben_Beng': 'ben', 'bos_Latn': 'bos', 'bul_Cyrl': 'bul', 'cat_Latn': 'cat', 'ceb_Latn': 'ceb', 'ces_Latn': 'ces', 'ckb_Arab': 'ckb', 'cym_Latn': 'cym', 'dan_Latn': 'dan', 'deu_Latn': 'deu', 'ell_Grek': 'ell', 'eng_Latn': 'eng', 'est_Latn': 'est', 'fin_Latn': 'fin', 'fra_Latn': 'fra', 'fuv_Latn': 'ful', 'gle_Latn': 'gle', 'glg_Latn': 'glg', 'guj_Gujr': 'guj', 'hau_Latn': 'hau', 'heb_Hebr': 'heb', 'hin_Deva': 'hin', 'hrv_Latn': 'hrv', 'hun_Latn': 'hun', 'hye_Armn': 'hye', 'ibo_Latn': 'ibo', 'ind_Latn': 'ind', 'isl_Latn': 'isl', 'ita_Latn': 'ita', 'jav_Latn': 'jav', 'jpn_Jpan': 'jpn', 'kam_Latn': 'kam', 'kan_Knda': 'kan', 'kat_Geor': 'kat', 'kaz_Cyrl': 'kaz', 'khm_Khmr': 'khm', 'kir_Cyrl': 'kir', 'kor_Hang': 'kor', 'lao_Laoo': 'lao', 'lij_Latn': 'Latvian', 'lim_Latn': 'kea', 'lin_Latn': 'lin', 'lit_Latn': 'lit', 'ltz_Latn': 'ltz', 'lug_Latn': 'lug', 'luo_Latn': 'luo', 'lvs_Latn': 'lav', 'mal_Mlym': 'mal', 'mar_Deva': 'mar', 'mkd_Cyrl': 'mkd', 'mlt_Latn': 'mlt', 'khk_Cyrl': 'mon', 'mri_Latn': 'mri', 'mya_Mymr': 'mya', 'nld_Latn': 'nld', 'nob_Latn': 'nob', 'npi_Deva': 'npi', 'nso_Latn': 'nso', 'nya_Latn': 'nya', 'oci_Latn': 'oci', 'gaz_Latn': 'orm', 'ory_Orya': 'ory', 'pan_Guru': 'pan', 'pes_Arab': 'fas', 'pol_Latn': 'pol', 'por_Latn': 'por', 'pbt_Arab': 'pus', 'ron_Latn': 'ron', 'rus_Cyrl': 'rus', 'slk_Latn': 'slk', 'sna_Latn': 'sna', 'snd_Arab': 'snd', 'som_Latn': 'som', 'spa_Latn': 'spa', 'srp_Cyrl': 'srp', 'swe_Latn': 'swe', 'swh_Latn': 'swh', 'tam_Taml': 'tam', 'tel_Telu': 'tel', 'tgk_Cyrl': 'tgk', 'tgl_Latn': 'tgl', 'tha_Thai': 'tha', 'tur_Latn': 'tur', 'ukr_Cyrl': 'ukr', 'umb_Latn': 'umb', 'urd_Arab': 'urd', 'uzn_Latn': 'uzb', 'vie_Latn': 'vie', 'wol_Latn': 'wol', 'xho_Latn': 'xho', 'yor_Latn': 'yor', 'zho_Hans': 'zho_simpl', 'zho_Hant': 'zho_trad', 'zsm_Latn': 'msa', 'zul_Latn': 'zul'}

# Mapping from all kinds of language names to the same standardized codes
LANGS_TO_FLORES_CODE = {}

for name, code in FLORES_LANGS.items():
    LANGS_TO_FLORES_CODE[name.lower()] = code
    LANGS_TO_FLORES_CODE[code.lower()] = code
    # This may lead to some incorrectly assigned scripts
    LANGS_TO_FLORES_CODE[code.split("_")[0]] = code
    for name2, code2 in FLORES_LANGS.items():
        if code == code2: continue
        #TRAIN_DATASETS.append(("facebook/flores", f"{code}-{code2}"))

ME_LANGUAGES = ["en", "da", "de", "nl", "sv", "bg", "cs", "hr", "pl", "sk", "sl", "es", "fr", "it", "pt", "ro", "et", "fi", "hu", "lt", "lv", "el", "mt"]
for l1 in ME_LANGUAGES:
    for l2 in ME_LANGUAGES:
        if l1 == l2: continue
        TRAIN_DATASETS.append(("Muennighoff/multi_eurlex", f"{l1}-{l2}"))

for new_code, old_code in FLORES_NEW_TO_OLD.items():
    LANGS_TO_FLORES_CODE[old_code] = new_code
    LANGS_TO_FLORES_CODE[new_code] = new_code

    try:
        name = languages.get(part3=old_code)
        LANGS_TO_FLORES_CODE[name.part1] = new_code
        LANGS_TO_FLORES_CODE[name.name.lower()] = new_code
        LANGS_TO_FLORES_CODE[name.name.lower().split(" ")[0]] = new_code
    except KeyError:
        print(f"Could not find iso3 code for {old_code}.")

# Add programming languages
LANGS_TO_FLORES_CODE["python"] = "py"
LANGS_TO_FLORES_CODE["javascript"] = "js"
LANGS_TO_FLORES_CODE["java"] = "java"
LANGS_TO_FLORES_CODE["cpp"] = "cpp"
LANGS_TO_FLORES_CODE["c"] = "c"
LANGS_TO_FLORES_CODE["go"] = "go"
LANGS_TO_FLORES_CODE["rust"] = "rust"

DS_TO_LANG = {
    "python": "python",
    'Muennighoff/mbpp': 'python',
    'openai_humaneval': 'python',
    "great_code": "python",
    "neural_code_search": "python",
    "codeparrot/codecomplex": "java",
    "codeparrot/github-jupyter-text-code-pairs": "jupyter-notebook",
    "codeparrot/apps": "python",
    "Fraser/python-state-changes": "python",
    "codeparrot/xlcost-text-to-code": "python",
    "teven/code_contests": "python",
    "teven/code_docstring_corpus": "python",
    "clue": "zho_Hans",
    "cmn": "zho_Hans", # == zho
    "cmn_Hans": "zho_Hans", # == zho
    "cmn_Hant": "zho_Hant", # == zho
    "zh": "zho_Hans", # == zho
    "jp": "jpn_Jpan", # == jpn
    "npi": "npi_Deva", # == npe
    "ory": "ory_Orya", # == ori
    "swh": "swh_Latn", # == swa
    "sw": "swh_Latn", # == swa
    "eu": "eus_Latn", # == eus
    "qu": "que_Latn", # == que
    "tr": "tur_Latn", # == tur
    "vi": "vie_Latn", # == vie
    "ta": "tam_Taml", # == tam
    "te": "tel_Telu", # == tel
    "th": "tha_Thai", # == tha
    "ht": "hat_Latn", # == hat
    "wuu": "wuu_Hans", # == wuu
    "yue_Hans": "yue_Hans", # == yue
    "wuu_Hans": "wuu_Hans", # == wuu
    "srp_Latn": "srp_Latn", # == srp
    "nor": "nor_Latn", # == Norwegian; Macro language
    "yid": "yid_Hebr", # Yiddish; Macro
    "tigrinya": "tir_Ethi", # == tir
    "kirundi": "run_Latn", # == rundi
    "punjabi": "pan_Guru", # == panjabi
    "chinese_simplified": "zho_Hans",
    "chinese_traditional": "zho_Hant",
    "chinese": "zho_Hans",
    "farsi": "pes_Arab",
    "bangla": "ben_Beng",
    "Ghanaian Pidgin English": "gpe_Latn",
    "python": "python",
    "castilian": "spa_Latn",
    "serbian_latin": "srp_Latn",
    "pashto": "pbt_Arab",
    "azerbaijani": "aze_Latn",
    "scottish_gaelic": "gla_Latn",
    "gaelic": "gla_Latn",
    "romano-serbian": "rsb_Latn",
    "sinhalese": "sin_Sinh",
    "serbian_cyrillic": "srp_Cyrl",
    "pidgin": "pcm_Latn",
    "kiswahili": "swh_Latn",
    "uighur": 'uig_Arab',
    "fur": "fur_Latn",
    "albanian": "sqi_Latn",
    "quechua": "quy_Latn",
    "Cornish": "cor_Latn",
    "flemish": "nld_Latn",
    "chuvash": "chv_Cyrl",
    "modern greek": "ell_Grek",
    "western frisian": "fry_Latn",
    "interlingua": "ina_Latn",
    "kurdish": "kur_Latn",
    "java": "java",
    ### Languages not in flores ###
    "ain": "ain_Latn",
    "ain_Latn": "ain_Latn",
    "ber": "ber_Latn",
    "ber_Latn": "ber_Latn",
    "ber_Tfng": "ber_Tfng",
    "ber_Arab": "ber_Arab",
    "arq": "arq_Arab",
    "arq_Arab": "arq_Arab",
    "avk": "avk_Latn",
    "avk_Latn": "avk_Latn",
    "awa": "awa_Deva",
    "awa_Deva": "awa_Deva",
    "aze": "aze_Latn",
    "aze_Latn": "aze_Latn",
    "bre": "bre_Latn",
    "bre_Latn": "bre_Latn",
    "bua": "bua_Cyrl",
    "bua_Cyrl": "bua_Cyrl",
    "cbk": "cbk_Latn",
    "cbk_Latn": "cbk_Latn",
    "cha": "cha_Latn",
    "cha_Latn": "cha_Latn",
    # They all intermingle Katakana/Hiragana/Kanji, but they are guaranteed to have the individual style; I.e. Kana is guaranteed to have katakana in each sample
    "jpn_Hira": "jpn_Hira",
    "jpn_Kana": "jpn_Kana",
    "jpn_Hani": "jpn_Hani",
    "lat": "lat_Latn",
    "lat_Latn": "lat_Latn",
    "dsb": "dsb_Latn",
    "dsb_Latn": "dsb_Latn",
    "fry": "fry_Latn",
    "fry_Latn": "fry_Latn",
    "hoc": "hoc_Latn",
    "hoc_Deva": "hoc_Deva",
    "hoc_Latn": "hoc_Latn",
    "frr": "frr_Latn",
    "frr_Latn": "frr_Latn",
    "jbo": "jbo_Latn",
    "jbo_Latn": "jbo_Latn",
    "tlh": "tlh_Latn",
    "tlh_Latn": "tlh_Latn",
    "lfn": "lfn_Latn",
    "lfn_Latn": "lfn_Latn",
    "lfn_Cyrl": "lfn_Cyrl",
    "vol": "vol_Latn",
    "vol_Latn": "vol_Latn",
    "tzl": "tzl_Latn",
    "tzl_Latn": "tzl_Latn",
    "gos": "gos_Latn",
    "gos_Latn": "gos_Latn",
    "hbs": "hbs_Latn",
    "hbs_Latn": "hbs_Latn",
    "hrx": "hrx_Latn",
    "hrx_Latn": "hrx_Latn",
    "hsb": "hsb_Latn",
    "hsb_Latn": "hsb_Latn",
    "xal": "xal_Cyrl",
    "xal_Cyrl": "xal_Cyrl",
    "toki": "toki_Latn",
    "toki_Latn": "toki_Latn",
    "tok_Latn": "tok_Latn",
    "sah": "sah_Cyrl",
    "sah_Cyrl": "sah_Cyrl",
    "kur_Latn": "kur_Latn",
    "ido": "ido_Latn",
    "ido_Latn": "ido_Latn",
    "kdr_Latn": "kdr_Latn",
    "kdr_Cyrl": "kdr_Cyrl",
    "kzj": "kzj_Latn",
    "kzj_Latn": "kzj_Latn",
    "lad_Latn": "lad_Latn",
    "ota_Arab": "ota_Arab",
    "ota_Latn": "ota_Latn",
    "uzb_Latn": "uzb_Latn",
    "chm": "chm_Cyrl",
    "chv": "chv_Cyrl",
    "cor": "cor_Latn",
    "dtp": "dtp_Latn",
    "egl": "egl_Latn",
    "fkv": "fkv_Latn",
    "gcf": "gcf_Latn",
    "got": "got_Goth",
    "grc": "grc_Grek",
    "gsw": "gsw_Latn",
    "ile": "ile_Latn",
    "ina": "ina_Latn",
    "ina_Latn": "ina_Latn",
    "kha": "kha_Latn",
    "kur": "kur_Latn",
    "lad": "lad_Latn",
    "nds": "nds_Latn",
    "nov": "nov_Latn",
    "nst": "nst_Latn",
    "orv": "orv_Cyrl",
    "ota": "ota_Arab",
    "pam": "pam_Latn",
    "pcd": "pcd_Latn",
    "pms": "pms_Latn",
    "prg": "prg_Latn",
    "que": "que_Latn",
    "rom": "rom_Latn",
    "sqi": "sqi_Latn",
    "swa": "swa_Latn",
    "swg": "swg_Latn",
    "zza": "zza_Latn",
    "sl": "slv_Latn",
    **LANGS_TO_FLORES_CODE,
    **{d: "eng_Latn" for (d,s) in TRAIN_DATASETS_EN},
}

# Add names
FLORES_LANGS_INV["uzb_Latn"] = "Uzbek (Latin script)"
FLORES_LANGS_INV["ota_Arab"] = "Ottoman Turkish"
FLORES_LANGS_INV["ota_Latn"] = "Ottoman Turkish (Latin script)"
FLORES_LANGS_INV["lad_Latn"] = "Ladino"
FLORES_LANGS_INV["kzj_Latn"] = "Coastal Kadazan"
FLORES_LANGS_INV["kdr_Latn"] = "Karaim (Latin script)"
FLORES_LANGS_INV["kdr_Cyrl"] = "Karaim (Cyrillic script)"
FLORES_LANGS_INV["ido_Latn"] = "Ido"
FLORES_LANGS_INV["kur_Latn"] = "Kurdish (Latin script)"
FLORES_LANGS_INV["yue_Hans"] = "Yue Chinese (Simplified)"
FLORES_LANGS_INV["sah_Cyrl"] = "Yakut"
FLORES_LANGS_INV["tok_Latn"] = "Toki Pona"
FLORES_LANGS_INV["toki_Latn"] = "Toki Pona"
FLORES_LANGS_INV["toki"] = "Toki Pona"
FLORES_LANGS_INV["xal"] = "Kalmyk"
FLORES_LANGS_INV["ain"] = "Ainu"
FLORES_LANGS_INV["ain_Latn"] = "Ainu (Latin script)"
FLORES_LANGS_INV["ber"] = "Berber"
FLORES_LANGS_INV["ber_Latn"] = "Berber (Latin script)"
FLORES_LANGS_INV["ber_Tfng"] = "Berber (Tifinagh script)"
FLORES_LANGS_INV["ber_Arab"] = "Berber (Arabic script)"
FLORES_LANGS_INV["arq_Arab"] = "Algerian Arabic"
FLORES_LANGS_INV["avk_Latn"] = "Kotava"
FLORES_LANGS_INV["awa_Deva"] = "Awadhi"
FLORES_LANGS_INV["aze_Latn"] = "Azerbaijani (South or North; Latin script)"
FLORES_LANGS_INV["bre_Latn"] = "Breton"
FLORES_LANGS_INV["bua_Cyrl"] = "Buryat"
FLORES_LANGS_INV["cbk_Latn"] = "Chavacano"
FLORES_LANGS_INV["cha_Latn"] = "Chamorro"
FLORES_LANGS_INV["jpn_Hira"] = "Japanese (Hiragana)"
FLORES_LANGS_INV["jpn_Kana"] = "Japanese (Katakana)"
FLORES_LANGS_INV["jpn_Hani"] = "Japanese (Kanji)"
FLORES_LANGS_INV["lat_Latn"] = "Latin"
FLORES_LANGS_INV["dsb_Latn"] = "Lower Sorbian"
FLORES_LANGS_INV["hsb_Latn"] = "Upper Sorbian"
FLORES_LANGS_INV["fry_Latn"] = "Frisian"
FLORES_LANGS_INV["hoc_Deva"] = "Ho (Devanagari script)"
FLORES_LANGS_INV["hoc_Latn"] = "Ho (Latin script)"
FLORES_LANGS_INV["frr_Latn"] = "Northern Frisian"
FLORES_LANGS_INV["jbo_Latn"] = "Lojban"
FLORES_LANGS_INV["nor_Latn"] = "Norwegian"
FLORES_LANGS_INV["yid_Hebr"] = "Yiddish"
FLORES_LANGS_INV["tlh_Latn"] = "Klingon"
FLORES_LANGS_INV["lfn_Latn"] = "Lingua Franca Nova"
FLORES_LANGS_INV["lfn_Cyrl"] = "Lingua Franca Nova (Cyrillic script)"
FLORES_LANGS_INV["vol_Latn"] = "Volapük"
FLORES_LANGS_INV["tzl_Latn"] = "Talossan"
FLORES_LANGS_INV["srp_Latn"] = "Serbian (Latin script)"
FLORES_LANGS_INV["gos_Latn"] = "Gronings"
FLORES_LANGS_INV["hbs_Latn"] = "Serbo-Croatian" # Macro
FLORES_LANGS_INV["hrx_Latn"] = "Hunsrik"
FLORES_LANGS_INV["ile_Latn"] = "Interlingue"
FLORES_LANGS_INV["ina_Latn"] = "Interlingua (International Auxiliary Language Association)"

# From https://github.com/Helsinki-NLP/LanguageCodes/blob/e2d30a81e2aba5cb6af2c45369433e4a295aa52c/iso639
with open("tatoebalangs.txt", "r") as f:
    for line in f.readlines()[1:]:
        parts = line.split("\t")
        code, name = parts[0], parts[-2]
        if code not in DS_TO_LANG:
            print(code, name)
            assert code
            DS_TO_LANG[code] = code
        if code not in FLORES_LANGS_INV:
            FLORES_LANGS_INV[code] = name
            assert code
            assert name

# Add all that's in FLORES_LANGS_INV but not in DS_TO_LANG
for code, name in FLORES_LANGS_INV.items():
    if code not in DS_TO_LANG:
        print(code, name)
        assert code
        DS_TO_LANG[code] = code
    if name not in DS_TO_LANG:
        DS_TO_LANG[name] = code

DS_TO_LANG["python"] = "python"
DS_TO_LANG = {k.lower(): v for k,v in DS_TO_LANG.items() if (("_" in v) or (v in ("python", "java", "jupyter-notebook")))}
assert "python" in DS_TO_LANG
# To create maps
"""
import json
with open("xp3x_name_to_code.json", "w") as f:
    json.dump(DS_TO_LANG, f, ensure_ascii=False)

import json
with open("xp3x_code_to_name.json", "w") as f:
    json.dump(FLORES_LANGS_INV, f, ensure_ascii=False)

print(DS_TO_LANG)
exit()
"""

### DATASET CREATION ###

# Copied from promptsource.utils
def removeHyphen(example):
    example_clean = {}
    for key in example.keys():
        if "-" in key:
            new_key = key.replace("-", "_")
            example_clean[new_key] = example[key]
        else:
            example_clean[key] = example[key]
    example = example_clean
    return example

def apply_template(dataset, template, strip_connection=True):
    def map_fn(ex):
        ex = removeHyphen(ex)
        try:
            inputs_and_targets = template.apply(
                ex, 
                strip_connection=strip_connection,
                truncate=True,
            )
        # Skip ValueError("Prompt did not produce an input and at least one target.")
        # which happens for some prompts with if else clauses based on inputs producing occasional
        # empty targets
        except ValueError as e:
            print(f"Skipping example {ex} because of error {e}")
            return {"inputs": "", "targets": ""}
        if len(inputs_and_targets) == 2:
            # Note that the signature changed in promptsource 
            # In 0.1.0 template.apply returned two strings; In >0.3.0 it retuns a str & list
            inputs, targets = inputs_and_targets
            if len(targets) > 1:
                # Safer to skip, as could be a bug
                print(f"Found targets longer than 1. Inputs: {inputs} ; Targets {targets}. Skipping.")
                return {"inputs": "", "targets": ""}
            targets = targets[0]
            return {"inputs": inputs, "targets": targets}
        # When template results in an empty example, template.apply returns [""]
        # Also, if the template gets split wrong, len can be > 2
        # We will filter these out later
        else:
            # inputs is a str by default & targets a str
            return {"inputs": "", "targets": ""}

    def filter_fn(ex):
        return len(ex["inputs"]) > 0 and len(ex["targets"]) > 0

    original_columns = dataset.column_names
    dataset = dataset.map(map_fn).filter(filter_fn)
    # map keeps original columns, remove them
    return dataset.remove_columns(set(original_columns) - {"inputs", "targets"})

def add_language_name_wikilingua(example):
    example["source_language_name"] = languages.get(alpha2=example["source_language"]).name
    example["target_language_name"] = languages.get(alpha2=example["target_language"]).name
    return example

def add_language_name_flores(example, subset_name):
    l1, l2 = subset_name.split("-")
    example["source_language_name"] = FLORES_LANGS_INV[l1]
    example["target_language_name"] = FLORES_LANGS_INV[l2]
    return example

def add_language_name_tatoeba(example, inv=False):
    l1, l2 = example["sourceLang"], example["targetlang"]
    try:
        l1 = languages.get(part3=l1).name
    except:
        try:
            l1 = languages.get(part3=l1.split("_")[0]).name
        except:
            l1 = FLORES_LANGS_INV[l1]
    try:
        l2 = languages.get(part3=l2).name
    except:
        try:
            l2 = languages.get(part3=l2.split("_")[0]).name
        except:
            l2 = FLORES_LANGS_INV[l2]

    if inv:
        example["source_language_name"] = l2
        example["target_language_name"] = l1
    else:
        example["source_language_name"] = l1
        example["target_language_name"] = l2
    return example

def add_language_name_wmt22(example, inv=False):
    l1, l2 = list(example["translation"].keys())
    lang1 = languages.get(part3=l1).name
    lang2 = languages.get(part3=l2).name
    if inv:
        example["source_language_name"] = lang2
        example["target_language_name"] = lang1
        example["source"] = example["translation"][l2]
        example["target"] = example["translation"][l1]
    else:
        example["source_language_name"] = lang1
        example["target_language_name"] = lang2
        example["source"] = example["translation"][l1]
        example["target"] = example["translation"][l2]
    return example

def filter_l1_l2_wikilingua(example, l1, l2):
    return example["source_language"] == l1 and example["target_language"] == l2

def filter_empty_solution_apps(example):
    return bool(example["solutions"])

def add_solution_apps(example):
    example["solution"] = random.choice(json.loads(example["solutions"]))
    return example

def clean_code_xlcost(example):
    clean_lines = []
    cur_indent = 0
    for line in example["code"].split("NEW_LINE"):
        cur_indent += line.count("INDENT")
        cur_indent -= line.count("DEDENT")
        line = line.replace("INDENT", "").replace("DEDENT", "")
        line = line.replace("STRNEWLINE", "\n")
        line = line.replace("TABSYMBOL", "\t")
        clean_lines.append("\t" * cur_indent + line.strip())
    example["code_clean"] = "\n".join(clean_lines)
    return example

def write_to_jsonl_hub(ds):

    ### GET DATASET & LANGUAGE ###
    ds_name, subset_name = ds
    is_wikilingua_cross_lingual = (ds_name == "GEM/wiki_lingua") and ("_") in subset_name
    
    lang_dir = DS_TO_LANG.get(ds_name.lower())
    if lang_dir is None:
        lang_dir = "unknown"
        if subset_name is not None:
            lang_dir = DS_TO_LANG.get(subset_name.lower(), None)
        if ds_name in ("facebook/flores", "Muennighoff/multi_eurlex"):
            lang_dir = DS_TO_LANG.get(subset_name.split("-")[-1].lower())
        elif ds_name == "Helsinki-NLP/tatoeba_mt":
            splitted = subset_name.replace("-inverted", "").split("-")
            if len(splitted) != 2: raise ValueError("Unexpected length: " + subset_name)
            l1, l2 = splitted
            if l1 == l2: return     
            if subset_name.endswith("-inverted"):
                lang_dir = DS_TO_LANG.get(l1.lower())
                l_check = l1.lower()     
            else:
                lang_dir = DS_TO_LANG.get(l2.lower())
                l_check = l2.lower()     

            if lang_dir is None:
                lang_dir = DS_TO_LANG.get(l_check.split("_")[0])
                if (lang_dir is not None) and (l_check.split("_")[1] == "latn"):
                    lang_dir += "_Latn"
                    FLORES_LANGS_INV[lang_dir] = FLORES_LANGS_INV[l_check.split("_")[0]] + " (Latin script)"
                elif (lang_dir is not None) and (l_check.split("_")[1] == "cyrl"):
                    lang_dir += "_Cyrl"
                    FLORES_LANGS_INV[lang_dir] = FLORES_LANGS_INV[l_check.split("_")[0]] + " (Cyrillic script)"
                elif (lang_dir is not None):
                    raise ValueError(f"Unknown script for {l_check}")
                else:
                    raise ValueError(f"Unknown language for {l_check}")
        elif ds_name == "allenai/wmt22_african":
            if subset_name.endswith("-inverted"):
                lang_dir = DS_TO_LANG.get(subset_name.split("-")[0].lower())
            else:
                lang_dir = DS_TO_LANG.get(subset_name.split("-")[1].lower())
        elif is_wikilingua_cross_lingual or ds_name == "pasinit/xlwic":
            lang_dir = DS_TO_LANG.get(subset_name.split("_")[-1].lower())
        elif ds_name == "xquad":
            lang_dir = DS_TO_LANG.get(subset_name.split(".")[1].lower())
        elif ds_name == "mlqa":
            # Classify it by the target language for cross-lingual (i.e. what the loss is computed on)
            lang_dir = DS_TO_LANG.get(subset_name.split(".")[1].lower())
        
        if (lang_dir is None):
            raise ValueError(f"Unknown language for {ds_name}/{subset_name}")

        print(f"Using {lang_dir} as language dir for {ds_name}/{subset_name}")
    
    os.makedirs(lang_dir, exist_ok=True)

    if ds_name == "Helsinki-NLP/tatoeba_mt":
        if subset_name.endswith("-inverted"):
            try:
                ds = load_dataset(ds_name, subset_name.replace("-inverted", ""), ignore_verifications=True)
            except:
                print(f"Failed to load {ds_name}/{subset_name.replace('-inverted', '')}")
                return
            ds = ds.map(lambda x: add_language_name_tatoeba(x, inv=True))
            ds = ds.rename_column(f"sourceString", "tmp")
            ds = ds.rename_column(f"targetString", f"sourceString")
            ds = ds.rename_column(f"tmp", f"targetString")
        else:
            # Sometimes has NonMatchingSplitsSizesError hence ignore
            try:
                ds = load_dataset(ds_name, subset_name, ignore_verifications=True)
            except:
                print(f"Failed to load {ds_name}/{subset_name}")
                return
            ds = ds.map(lambda x: add_language_name_tatoeba(x, inv=False))
    elif ds_name == "allenai/wmt22_african":
        if subset_name.endswith("-inverted"):
            ds = load_dataset(ds_name, subset_name.replace("-inverted", ""))
            ds = ds.map(lambda x: add_language_name_wmt22(x, inv=True))
        else:
            ds = load_dataset(ds_name, subset_name)
            ds = ds.map(lambda x: add_language_name_wmt22(x, inv=False))
    elif ds_name == "story_cloze":
        ds = load_dataset(ds_name, subset_name, data_dir=STORY_CLOZE_DIR)
    elif ds_name == "Muennighoff/xstory_cloze":
        ds = load_dataset(ds_name, subset_name, data_dir=XSTORY_CLOZE_DIR)
    else:
        ds = load_dataset(ds_name, subset_name)

    if ds_name == "GEM/wiki_lingua":
        # Add names, e.g. Chinese for zh to use them in the jinja prompts
        ds = ds.map(add_language_name_wikilingua)
        if is_wikilingua_cross_lingual:
            # Keep only L1 -> L2 (L2 -> L1 will be a separate dataset)
            ds = ds.filter(partial(filter_l1_l2_wikilingua, l1=subset_name.split("_")[0], l2=subset_name.split("_")[1]))
    elif ds_name == "facebook/flores":
        ds = ds.map(lambda x: add_language_name_flores(x, subset_name))
        l1, l2 = subset_name.split("-")
        ds = ds.rename_column(f"sentence_{l1}", "source")
        ds = ds.rename_column(f"sentence_{l2}", "target")
    elif ds_name == "codeparrot/apps":
        ds = ds.filter(filter_empty_solution_apps).map(add_solution_apps)
    elif ds_name == "codeparrot/xlcost-text-to-code":
        ds = ds.map(clean_code_xlcost)

    ### SELECT SPLITS ###
    dataset_splits = list(ds.keys())
    if subset_name.startswith("xlwic_en_"):
        # Train set is en; val & test are zh
        dataset_splits.remove("train")
    elif ds_name == "teven/code_docstring_corpus":
        # Bad quality split
        dataset_splits.remove("class_level")
    elif ds_name == "GEM/wiki_lingua":
        # Remove samples
        dataset_splits.remove("sampled_validation")
        dataset_splits.remove("sampled_test")

    ### SELECT PROMPTS ###
    if subset_name is None:
        prompt_dataset_name = ds_name
    else:
        subset_name_prompt = subset_name
        if USE_ENGLISH_PROMPTS and ds_name in DS_TO_ENG_PROMPT:
            subset_name_prompt = DS_TO_ENG_PROMPT[ds_name]
        prompt_dataset_name = f"{ds_name}/{subset_name_prompt}"

    prompts = DatasetTemplates(prompt_dataset_name)

    ### PROCESS ###

    for split in dataset_splits:
        for t_name in prompts.all_template_names:
            print(f"Running {ds_name}/{subset_name}/{split}/{t_name}")
            if SKIP_PROMPTS.get(prompt_dataset_name, {}).get(split, False):
                if ("all" in SKIP_PROMPTS[prompt_dataset_name][split]) or (t_name in SKIP_PROMPTS[prompt_dataset_name][split]):
                    print(f"Skipping DS: {prompt_dataset_name} Split {split} Prompt {t_name}")
                    continue
            
            out_path = os.path.join(
                lang_dir, 
                f'xp3_{ds_name}_{subset_name}_{split}_{t_name}.jsonl'.replace("/", "_").replace(" ", "_").replace("-", "_")
            )
            if os.path.exists(out_path):
                print("Skipping as exists: ", out_path)
                continue
            
            assert len(ds[split]) > 0, f"Got empty: {ds_name}"

            try:
                if ds_name == "allenai/wmt22_african":
                    # Sort by laser score, i.e. by increasing confidence & limit samples due to mediocre quality
                    ds[split] = ds[split].sort("laser_score", reverse=True)
                    max_range = min(len(ds[split]), MAX_EXAMPLES_PER_DATASET_PROMPT // 2)
                else:
                    # Allow 5x buffer for empty examples
                    max_range = min(len(ds[split]), MAX_EXAMPLES_PER_DATASET_PROMPT * 5)
                # Shuffle to avoid using the same subset
                # Leave \n in-between input & targets for code
                out_ds = apply_template(
                    dataset=ds[split].shuffle().select(list(range(max_range))), 
                    template=prompts[t_name],
                    strip_connection=False if lang_dir == "code" else True
                )
                # Keep X shortest examples
                max_range = min(len(out_ds), MAX_EXAMPLES_PER_DATASET_PROMPT)
                out_ds = out_ds.sort("inputs").select(list(range(max_range)))
            except Exception as e:
                print(f"Skipping due to {e}. DS: {ds_name}/{subset_name} Template: {t_name}")
                continue
            # Do not force ascii to allow chars like é
            if len(out_ds) > 0:
                def add_cols(example, keys, values):
                    for key, value in zip(keys, values):
                        example[key] = value
                    return example
                out_ds = out_ds.map(
                    lambda x: add_cols(
                        x, 
                        ["language", "split", "template", "dataset", "config"], 
                        [lang_dir, split, t_name, ds_name, subset_name]
                    )
                )
                out_ds.to_json(out_path, orient="records", lines=True, force_ascii=False)
            else:
                print(f"Skipping due to empty. DS: {ds_name}/{subset_name} Template: {t_name}")

# Testing:
#TRAIN_DATASETS = [
#    ('xquad', 'xquad.ar'),
#]

#print(DS_TO_LANG)

#for ds in TRAIN_DATASETS:
#    write_to_jsonl_hub(ds)
    
if __name__ == "__main__":
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(write_to_jsonl_hub, TRAIN_DATASETS)


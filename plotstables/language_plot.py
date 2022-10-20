"""
Plot with languages across x-axis & percentage for ROOTS, mC4 & xP3 on y-axis
"""

# ISO Tokens Pages mT5
# Code Language (B) (M) (%) Code Language (B) (M) (%)
MC4 = """
en English 2,733 3,067 5.67 mk Macedonian 1.8 2.1 0.62
ru Russian 713 756 3.71 ml Malayalam 1.8 2.1 0.62
es Spanish 433 416 3.09 mn Mongolian 2.7 2.1 0.62
de German 347 397 3.05 ur Urdu 2.4 1.9 0.61
fr French 318 333 2.89 be Belarusian 2.0 1.7 0.59
it Italian 162 186 2.43 la Latin 1.3 1.7 0.58
pt Portuguese 146 169 2.36 eu Basque 1.4 1.6 0.57
pl Polish 130 126 2.15 tg Tajik 1.4 1.3 0.54
nl Dutch 73 96 1.98 te Telugu 1.3 1.2 0.52
tr Turkish 71 88 1.93 fy WestFrisian 0.4 1.1 0.51
ja Japanese 164 87 1.92 kn Kannada 1.1 1.1 0.51
vi Vietnamese 116 79 1.87 ky Kyrgyz 1.0 1.0 0.50
id Indonesian 69 70 1.80 sw Swahili 1.0 1.0 0.50
cs Czech 63 60 1.72 so Somali 1.4 0.9 0.48
zh Chinese 39 55 1.67 my Burmese 0.9 0.8 0.47
fa Persian 52 54 1.67 uz Uzbek 0.9 0.8 0.46
ar Arabic 57 53 1.66 km Khmer 0.6 0.8 0.46
sv Swedish 45 49 1.61 - Russian(Latin) 0.9 0.7 0.46
ro Romanian 52 46 1.58 sd Sindhi 1.6 0.7 0.45
el Greek 43 42 1.54 gu Gujarati 0.8 0.6 0.43
uk Ukrainian 41 39 1.51 - Hindi(Latin) 0.6 0.6 0.43
hu Hungarian 39 37 1.48 jv Javanese 0.3 0.6 0.42
da Danish 29 29 1.38 zu Zulu 0.2 0.6 0.42
fi Finnish 25 27 1.35 si Sinhala 0.8 0.5 0.41
no Norwegian 27 25 1.33 - Japanese(Latin) 0.3 0.5 0.41
bg Bulgarian 22 23 1.29 eo Esperanto 0.7 0.5 0.40
hi Hindi 24 19 1.21 co Corsican 0.2 0.5 0.40
sk Slovak 18 18 1.19 ga Irish 0.5 0.5 0.40
ko Korean 26 16 1.14 - Greek(Latin) 0.4 0.4 0.39
th Thai 11 15 1.14 - Chinese(Latin) 0.2 0.4 0.37
ca Catalan 13 14 1.12 pa Punjabi 0.6 0.4 0.37
ms Malay 13 13 1.09 ceb Cebuano 0.2 0.4 0.36
iw Hebrew 17 12 1.06 mg Malagasy 0.2 0.3 0.36
lt Lithuanian 11 11 1.04 ps Pashto 0.4 0.3 0.36
sl Slovenian 8.8 8.5 0.95 sn Shona 0.2 0.3 0.35
mr Marathi 14 7.8 0.93 gd ScottishGaelic 0.4 0.3 0.35
bn Bengali 7.3 7.4 0.91 ku Kurdish 0.4 0.3 0.34
et Estonian 6.9 6.9 0.89 hmn Hmong 0.2 0.3 0.34
lv Latvian 7.0 6.4 0.87 su Sundanese 0.1 0.3 0.34
az Azerbaijani 4.4 5.3 0.82 ht HaitianCreole 0.2 0.3 0.33
gl Galician 2.4 4.6 0.79 ha Hausa 0.2 0.2 0.33
cy Welsh 4.9 4.1 0.76 ny Chichewa 0.1 0.2 0.29
sq Albanian 4.0 4.1 0.76 am Amharic 0.3 0.2 0.29
ta Tamil 3.4 3.5 0.73 - Bulgarian(Latin) 0.09 0.2 0.29
sr Serbian 4.3 3.4 0.72 yi Yiddish 0.3 0.1 0.28
ne Nepali 3.2 2.9 0.69 lo Lao 0.1 0.1 0.28
lb Luxembourgish 1.0 2.7 0.68 mi Maori 0.1 0.1 0.25
hy Armenian 2.4 2.4 0.65 sm Samoan 0.09 0.1 0.25
kk Kazakh 3.1 2.4 0.65 ig Igbo 0.09 0.09 0.24
ka Georgian 2.5 2.3 0.64 haw Hawaiian 0.09 0.08 0.24
mt Maltese 5.2 2.3 0.64 xh Xhosa 0.06 0.07 0.22
af Afrikaans 1.7 2.2 0.63 st Sotho 0.08 0.07 0.22
fil Filipino 2.1 2.1 0.62 yo Yoruba 0.05 0.05 0.20
is Icelandic 2.6 2.1 0.62 is Icelandic 2.6 2.1 0.62
"""

# Language ISO-639-3 catalog-ref Genus Family Macroarea Size in Bytes
ROOTS = """
Akan                & aka       & ak          & Kwa                     & Niger-Congo    & Africa    & 70,1554        \\
Arabic              & arb       & ar          & Semitic                 & Afro-Asiatic   & Eurasia   & 74,854,900,600   \\
Assamese            & asm       & as          & Indic                   & Indo-European  & Eurasia   & 291,522,098     \\
Bambara             & bam       & bm          & Western Mande           & Mande          & Africa    & 391,747        \\
Basque              & eus       & eu          & Basque                  & Basque         & Eurasia   & 2,360,470,848    \\
Bengali             & ben       & bn          & Indic                   & Indo-European  & Eurasia   & 18,606,823,104   \\
Catalan             & cat       & ca          & Romance                 & Indo-European  & Eurasia   & 17,792,493,289   \\
Chi Chewa           & nya       & ny          & Bantoid                 & Niger-Congo    & Africa    & 1,187,405       \\
Chi Shona           & sna       & sn          & Bantoid                 & Niger-Congo    & Africa    & 6,638,639       \\
Chi Tumbuka         & tum       & tum         & Bantoid                 & Niger-Congo    & Africa    & 170,360        \\
English             & eng       & en          & Germanic                & Indo-European  & Eurasia   & 484,953,009,124  \\
Fon                 & fon       & fon         & Kwa                     & Niger-Congo    & Africa    & 2,478,546       \\
French              & fra       & fr          & Romance                 & Indo-European  & Eurasia   & 208,242,620,434  \\
Gujarati            & guj       & gu          & Indic                   & Indo-European  & Eurasia   & 1,199,986,460    \\
Hindi               & hin       & hi          & Indic                   & Indo-European  & Eurasia   & 24,622,119,985   \\
Igbo                & ibo       & ig          & Igboid                  & Niger-Congo    & Africa    & 14078,521      \\
Indonesian          & ind       & id          & Malayo-Sumbawan         & Austronesian   & Papunesia & 19,972,325,222   \\
Isi Zulu            & zul       & zu          & Bantoid                 & Niger-Congo    & Africa    & 8,511,561       \\
Kannada             & kan       & kn          & Southern Dravidian      & Dravidian      & Eurasia   & 2,098,453,560    \\
Kikuyu              & kik       & ki          & Bantoid                 & Niger-Congo    & Africa    & 359,615        \\
Kinyarwanda         & kin       & rw          & Bantoid                 & Niger-Congo    & Africa    & 40,428,299      \\
Kirundi             & run       & rn          & Bantoid                 & Niger-Congo    & Africa    & 3,272,550       \\
Lingala             & lin       & ln          & Bantoid                 & Niger-Congo    & Africa    & 1,650,804       \\
Luganda             & lug       & lg          & Bantoid                 & Niger-Congo    & Africa    & 4,568,367       \\
Malayalam           & mal       & ml          & Southern Dravidian      & Dravidian      & Eurasia   & 3,662,571,498    \\
Marathi             & mar       & mr          & Indic                   & Indo-European  & Eurasia   & 1,775,483,122    \\
Nepali              & nep       & ne          & Indic                   & Indo-European  & Eurasia   & 2,551,307,393    \\
Northern Sotho      & nso       & nso         & Bantoid                 & Niger-Congo    & Africa    & 1,764,506       \\
Odia                & ori       & or          & Indic                   & Indo-European  & Eurasia   & 1,157,100,133    \\
Portuguese          & por       & pt          & Romance                 & Indo-European  & Eurasia   & 79,277,543,375   \\
Punjabi             & pan       & pa          & Indic                   & Indo-European  & Eurasia   & 1,572,109,752    \\
Sesotho             & sot       & st          & Bantoid                 & Niger-Congo    & Africa    & 751,034        \\
Setswana            & tsn       & tn          & Bantoid                 & Niger-Congo    & Africa    & 1,502,200       \\
Simplified Chinese  &     ---      & zhs         & Chinese                 & Sino-Tibetan   & Eurasia   & 261,019,433,892  \\
Spanish             & spa       & es          & Romance                 & Indo-European  & Eurasia   & 175,098,365,045  \\
Swahili             & swh       & sw          & Bantoid                 & Niger-Congo    & Africa    & 236,482,543     \\
Tamil               & tam       & ta          & Southern Dravidian      & Dravidian      & Eurasia   & 7,989,206,220    \\
Telugu              & tel       & te          & South-Central Dravidian & Dravidian      & Eurasia   & 2993407,159    \\
Traditional Chinese &      ---     & zht         & Chinese                 & Sino-Tibetan   & Eurasia   & 762,489,150     \\
Twi                 & twi       & tw          & Kwa                     & Niger-Congo    & Africa    & 1,265,041       \\
Urdu                & urd       & ur          & Indic                   & Indo-European  & Eurasia   & 2,781,329,959    \\
Vietnamese          & vie       & vi          & Viet-Muong              & Austro-Asiatic & Eurasia   & 43,709,279,959   \\
Wolof               & wol       & wo          & Wolof                   & Niger-Congo    & Africa    & 3,606,973       \\
Xhosa               & xho       & xh          & Bantoid                 & Niger-Congo    & Africa    & 14,304,074      \\
Xitsonga            & tso       & ts          & Bantoid                 & Niger-Congo    & Africa    & 707,634        \\
Yoruba              & yor       & yo          & Defoid                  & Niger-Congo    & Africa    & 89,695,835      \\
Programming Languages  & code      & code        & ---                    & ---           &       & 174,700,245,772 \\
"""


XP3 = """
106288	tw/merged_tw.jsonl
107056	bm/merged_bm.jsonl
108096	ak/merged_ak.jsonl
108112	eu/merged_eu.jsonl
110608	ca/merged_ca.jsonl
113072	fon/merged_fon.jsonl
114080	st/merged_st.jsonl
115040	ki/merged_ki.jsonl
116032	tum/merged_tum.jsonl
122560	wo/merged_wo.jsonl
126304	ln/merged_ln.jsonl
156256	as/merged_as.jsonl
161472	or/merged_or.jsonl
165456	kn/merged_kn.jsonl
175040	ml/merged_ml.jsonl
192992	rn/merged_rn.jsonl
229712	nso/merged_nso.jsonl
235536	tn/merged_tn.jsonl
235936	lg/merged_lg.jsonl
249360	rw/merged_rw.jsonl
250256	ts/merged_ts.jsonl
252496	sn/merged_sn.jsonl
254672	xh/merged_xh.jsonl
263712	zu/merged_zu.jsonl
272128	ny/merged_ny.jsonl
325232	ig/merged_ig.jsonl
352784	yo/merged_yo.jsonl
393680	ne/merged_ne.jsonl
523248	pa/merged_pa.jsonl
560688	gu/merged_gu.jsonl
560896	sw/merged_sw.jsonl
666240	mr/merged_mr.jsonl
832720	bn/merged_bn.jsonl
924496	ta/merged_ta.jsonl
1332912	te/merged_te.jsonl
1918272	ur/merged_ur.jsonl
3101408	vi/merged_vi.jsonl
4330752	code/merged_code.jsonl
4393696	hi/merged_hi.jsonl
4589904	zh/merged_zh.jsonl
4606288	id/merged_id.jsonl
4677264	ar/merged_ar.jsonl
5546688	fr/merged_fr.jsonl
6129584	pt/merged_pt.jsonl
7571808	es/merged_es.jsonl
37261104	en/merged_en.jsonl
"""


HEAD = "Language & Code & ROOTS (perc) & xP3 (perc) & xP3 (MB) & xP3 (M tokens)\\\\\n"
ONE_LINE = "{} & {} & {} & {} & {} & {}\\\\\n"


MC4_PCT_DICT = {}
MC4_BTS_DICT = {}
MC4_SUM = 0

for line in MC4.split("\n")[1:-1]:
    print(line.split(" "))
    code_1, _, bts_1, _, pct_1, code_2, _, bts_2, _, pct_2 = line.split(" ")
    MC4_PCT_DICT[code_1] = float(pct_1)
    MC4_PCT_DICT[code_2] = float(pct_2)
    # TODO: Use Bytes isntead to compute %?
    MC4_BTS_DICT[code_1] = float(bts_1.replace(",", ""))
    MC4_BTS_DICT[code_2] = float(bts_2.replace(",", ""))


for k in {'tn', 'ts', 'ak', 'rn', 'tum', 'bm', 'tw', 'rw', 'nso', 'lg', 'as', 'or', 'wo', 'ln', 'ki', 'code', 'fon'}:
    MC4_PCT_DICT[k] = 0
    MC4_BTS_DICT[k] = 0

ROOTS_DICT = {}
ROOTS_BTS_DICT = {}

ROOTS_SUM = 0
for line in ROOTS.split("\n")[1:-1]:
    lang_name, _, code, _, _, _, bts = line.split("&")
    #lang_name = lang_name.strip("\t").strip(" ")
    code = code.strip("\t").strip(" ")
    bts = int(bts.strip("\\").strip("\t").strip(" ").replace(',', ''))
    ROOTS_SUM += bts
    ROOTS_DICT[code] = bts
    ROOTS_BTS_DICT[code] = bts

ROOTS_DICT["zh"] = ROOTS_DICT["zht"] + ROOTS_DICT["zhs"]
ROOTS_BTS_DICT["zh"] = ROOTS_BTS_DICT["zht"] + ROOTS_BTS_DICT["zhs"]
ROOTS_DICT.pop("zht")
ROOTS_DICT.pop("zhs")
ROOTS_BTS_DICT.pop("zht")
ROOTS_BTS_DICT.pop("zhs")

ROOTS_DICT = {k: (v / ROOTS_SUM) * 100 for k, v in ROOTS_DICT.items()}


XP3_DICT = {}
XP3_BTS_DICT = {}

XP3_SUM = 0
for line in XP3.split("\n")[1:-1]:
    bts, code = line.split("\t")
    #code = code.replace("xp3/", "")
    #code = code.replace("xp3/", "")
    code = code.split("/")[0]
    bts = int(bts)
    XP3_SUM += bts
    XP3_DICT[code] = bts
    XP3_BTS_DICT[code] = bts

XP3_DICT = {k: (v / XP3_SUM) * 100 for k, v in XP3_DICT.items()}

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(36,6))

langs = sorted(XP3_DICT, key=lambda k: XP3_DICT[k], reverse=True)

# Info on percentages
print({l: ROOTS_DICT[l] for l in langs})
print({l: XP3_DICT[l] for l in langs})

#x_axis = np.array(list(range(0, int(len(langs) * 1), 1)))
x_axis = np.arange(len(langs))
x_axis = np.array([i + 1*i for i in x_axis])

# https://coolors.co/palette/8ecae6-219ebc-023047-ffb703-fb8500

ax.bar(x_axis - 0.6, [XP3_DICT[k] for k in langs], width=0.6, label = 'xP3', color="#219EBC")
ax.bar(x_axis,  [ROOTS_DICT[k] for k in langs], width=0.6, label = 'ROOTS', color="#023047")
ax.bar(x_axis + 0.6, [MC4_PCT_DICT[k] for k in langs], width=0.6, label = 'mC4', color="#FB8500")

# ax2.plot(x_axis - 0.6, [XP3_BTS_DICT[k] for k in langs], marker="o", label = 'xP3 Bytes', color="#8ECAE6")

ax.set_xticks(x_axis, langs)

ax.set_yscale('log')
ax.set_yticks([25, 5, 1, 0.1, 0.01, 0.001, 0.0001])

ax.set_ylabel("% of corpus")

# ax2.set_yscale('log')
# ax2.set_ylabel('Bytes in xP3')

ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter()) # Allow non-mathematical values
ax.get_yaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.4f')) # Allow 4 digits after comma
ax.get_yaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g')) # Rmv trailing zeroes

lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, fontsize=15, ncol=2)
# lines2, labels2 = ax2.get_legend_handles_labels()
#ax.legend(lines + lines2, labels + labels2, fontsize=15, ncol=2)


plt.savefig('language_plot.png', dpi=300, bbox_inches='tight')

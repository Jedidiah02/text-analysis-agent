from text_agent import word_count, sentence_count, top_keywords

TEXT = (
    "SynapseBug is a Flask web app that summarizes text. It uses spaCy for "
    "sentence splitting and RAKE for keyword extraction. Users register and log in. "
    "Admins can delete users."
)


def test_word_count():
    assert word_count.invoke({"text": TEXT}) == 29


def test_sentence_count():
    assert sentence_count.invoke({"text": TEXT}) == 4


def test_top_keywords_most_frequent_first():
    result = top_keywords.invoke({"text": TEXT, "n": 3})
    assert result[0] == "users"
    assert len(result) == 3


def test_empty_text():
    assert word_count.invoke({"text": ""}) == 0
    assert sentence_count.invoke({"text": ""}) == 0
    assert top_keywords.invoke({"text": ""}) == []
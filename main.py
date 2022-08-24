from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Okt
from multiprocessing import Process, Queue
from itertools import combinations

okt = Okt()


def token(s, res):
    res.put((s, ' '.join(okt.nouns(s))))
    return


if __name__ == '__main__':
    result = Queue()
    jobs = []
    li = ['한끼 당근 1개', '친환경 당근 500g', '[KF365] 김구원선생 국내산 무농약 콩나물 300g', '[KF365] 애호박 1개', '[KF365] 팽이버섯 2입', '깐대파 500g',
          '[KF365] GAP 밀양 깻잎 3속', '[팜에이트] 무농약 간편 샐러드 5종', '[KF365] 파프리카 2입', '친환경 양파 1kg', '[KF365] 깐마늘 200g',
          '[KF365] 부추 200g', '[KF365] 다다기오이 3입', '[KF365] 무 1통', '[KF365] 가지 2입', '[KF365] 감자 1kg',
          '[KF365] 새송이버섯 400g', '양상추 1입', '[KF365] 흙대파 1단 1kg', 'GAP 오이 2입', '[KF365] 청양고추 200g']
    li2 = []
    ans = []

    for i, s in enumerate(li):
        jobs.append(Process(target=token, args=(s, result)))
        jobs[i].start()

    for i, s in enumerate(li):
        jobs[i].join()

    result.put('STOP')
    total = 0
    while True:
        tmp = result.get()
        if tmp == 'STOP':
            break
        else:
            li2.append(tmp)

    combs = combinations(li2, 2)

    for sent in combs:
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform((sent[0][1], sent[1][1]))
        idf = tfidf_vectorizer.idf_

        d = str(dict(zip(tfidf_vectorizer.get_feature_names_out(), idf)))
        r = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        ans.append((r, d, sent))

    ans.sort(reverse=True)

    for t in ans:
        print(f'Origin: [{t[2][0][0]}] vs [{t[2][1][0]}]')
        print(f'Tokenization: [{t[2][0][1]}] vs [{t[2][1][1]}]')
        print(f'Vector: {t[1]}')
        print(f'Similarity: {t[0] * 100:.2f}%')
        print()

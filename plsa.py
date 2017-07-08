# -*- coding:utf-8 -*-
import numpy as np
import warnings;warnings.filterwarnings('ignore') # numpyのwarningを無視

class PLSA():
    def __init__(self, N, Z):
        self.N = N
        self.X = N.shape[0]
        self.Y = N.shape[1]
        self.Z = Z

        self.Pz = np.random.rand(self.Z) # P(z)
        self.Px_z = np.random.rand(self.Z, self.X) # P(x|z)
        self.Py_z = np.random.rand(self.Z, self.Y) # P(y|z)

        # 正規化
        self.Pz /= np.sum(self.Pz)
        self.Px_z /= np.sum(self.Px_z, axis=1)[:, None]
        self.Py_z /= np.sum(self.Py_z, axis=1)[:, None]
    
    # 学習
    def train(self, k=1000, t=1.0e-7):
        prev_llh = 0
        for i in range(k):
            self.e_step()
            self.m_step()
            llh = self.llh()
            if abs((llh - prev_llh) / prev_llh) < t:
                break
            prev_llh = llh

    # Eステップ
    def e_step(self):
        self.Pz_xy = self.Pz[None, None, :] * self.Px_z.T[:, None, :] * self.Py_z.T[None, :, :]
        self.Pz_xy[np.isinf(self.Pz_xy)] = 0
        self.Pz_xy[np.isnan(self.Pz_xy)] = 0
        self.Pz_xy /= np.sum(self.Pz_xy, axis=2)[:, :, None]

    # Mステップ
    def m_step(self):
        NP = self.N[:, :, None] * self.Pz_xy
        # infとnanは0へ修正
        NP[np.isinf(NP)] = 0
        NP[np.isnan(NP)] = 0

        self.Pz = np.sum(NP, axis=(0, 1))
        self.Px_z = np.sum(NP, axis=1).T
        self.Py_z = np.sum(NP, axis=0).T
        
        self.Pz /= np.sum(self.Pz)
        self.Px_z /= np.sum(self.Px_z, axis=1)[:, None]
        self.Py_z /= np.sum(self.Py_z, axis=1)[:, None]

    # 対数尤度
    def llh(self):
        Pxy = self.Pz[None, None, :] * self.Px_z.T[:, None, :] * self.Py_z.T[None, :, :]
        Pxy = np.sum(Pxy, axis=2)
        lPxy = np.log(Pxy)
        lPxy[np.isinf(lPxy)] = 0
        return np.sum(self.N * lPxy)

    # perplexityを算出
    def perplexity(self, M):
        Pxy = self.Pz[None, None, :] * self.Px_z.T[:, None, :] * self.Py_z.T[None, :, :]
        Pxy = np.sum(Pxy, axis=2)
        lPxy = np.log2(Pxy)
        lPxy[np.isinf(lPxy)] = 0
        lPxy = np.sum(M * lPxy)
        lPxy = - lPxy / M.sum()
        return 2 ** lPxy
        
    # クラスごとの単語をファイルへ出力
    def topword(self, row, col, num):
        with open('claster_result.txt', 'w') as f:
            for z in range(self.Z):
                x = sorted(enumerate(self.Px_z.T), key=lambda x:x[1][z], reverse=True)[:10]
                y = sorted(enumerate(self.Py_z.T), key=lambda x:x[1][z], reverse=True)[:10]
                f.write('-----クラスタ:' + str(z) + '-----\n')
                for i in range(num):
                    f.write(row[x[i][0]] + ',\t' + col[y[i][0]] + '\n')

# 共起行列を作成
def create_matrix(lefts, rights, pairs):
    mat = np.zeros((len(lefts), len(rights)))
    for x, y in pairs:
        mat[lefts.index(x)][rights.index(y)] += 1
    return mat

def main():
    # クラスタの単語を調べるための辞書
    # クラスタの上位単語を確率値から単語を逆引きするために使用
    row, col = {}, {}
    # 1語目,2語目のボキャブラリ
    lefts, rights = set(), set()
    # 単語のペアを保存
    pairs_train, pairs_test = [], []
    
    file_train = 'train_nvpair_en.txt'
    file_test = 'test_nvpair_en.txt'

    with open(file_train, 'r') as txt:
        for t in txt:
            word = t.split()
            lefts.add(word[0])
            rights.add(word[1])
            pairs_train.append(word)
    lefts = list(lefts)
    rights = list(rights)

    for i, e in enumerate(lefts):
        row[i] = e
    for i, e in enumerate(rights):
        col[i] = e

    with open(file_test, 'r') as txt:
        for t in txt:
            word = t.split()
            pairs_test.append(word)

    # train, testそれぞれに対し共起回数の配列を作成
    N = create_matrix(lefts, rights, pairs_train)
    M = create_matrix(lefts, rights, pairs_test)

    plsa = PLSA(N, 10)
    print('初期値の対数尤度:')
    print(plsa.llh())

    plsa.train()

    print('訓練後の対数尤度:')
    print(plsa.llh())
    print('perplexity of training')
    print(plsa.perplexity(N))
    print('perplexity of test')
    print(plsa.perplexity(M))
    
    plsa.topword(row, col, 10)

if __name__ == "__main__":
    main()

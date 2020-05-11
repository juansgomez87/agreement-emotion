import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, Normalizer, OneHotEncoder
from sklearn import svm, model_selection, mixture
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputRegressor
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, jaccard_score
from skmultilearn.model_selection import IterativeStratification
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import sys

import pdb
import warnings


class SVMClassifier():
    def __init__(self,
                 language,
                 quadrant,
                 filter,
                 num_comp):
        self.ratings = pd.read_csv('data_ratings.all.csv')
        # self.annotations = pd.read_csv('annotations.csv')
        self.annotations = pd.read_csv('annotations_5s.csv')
        # self.annotations = pd.read_csv('annotations_10s.csv')
        self.pca_plot = False
        self.plot_anno_flag = True
        self.plot_hist = False
        self.multi_label = True
        self.seed = np.random.seed(1987)
        self.num_comp = num_comp
        self.list_feats_files = self.annotations.Feats.tolist()
        self.list_quad = np.unique(self.annotations.Quads).tolist()
        self.list_emo = np.unique(self.annotations.Emotion).tolist()
        self.list_songs = np.unique(self.ratings.excerpt).tolist()
        # select ratings from a given survey
        if language == 'e':
            self.ratings = self.ratings[self.ratings['language'] == 'english']
            print('Working with English surveys and {} ratings'.format(self.ratings.shape[0]))
        elif language == 'm':
            self.ratings = self.ratings[self.ratings['language'] == 'mandarin']
            print('Working with Mandarin surveys and {} ratings'.format(self.ratings.shape[0]))
        elif language == 's':
            self.ratings = self.ratings[self.ratings['language'] == 'spanish']
            print('Working with Spanish surveys and {} ratings'.format(self.ratings.shape[0]))
        elif language == 'g':
            self.ratings = self.ratings[self.ratings['language'] == 'german']
            print('Working with German surveys and {} ratings'.format(self.ratings.shape[0]))
        elif language == 'a':
            print('Working with All surveys and {} ratings'.format(self.ratings.shape[0]))
        else:
            print('Options for language are [e, m, s, g, a]!')
            sys.exit(0)

        if quadrant == 'y':
            self.anno = self.annotations.Quads.tolist()
            self.group_rat = True
        elif quadrant == 'n':
            self.anno = self.annotations.Emotion.tolist()
            self.group_rat = False
        else:
            print('Options for quadrant are [y] or [n]!')
            sys.exit(0)

        self.fid = self.annotations.index.tolist()

        self.X, self.y, self.y_oe = self.get_all_feats()

        self.y_all_anno_enc = self.update_labels(None)
        self.y_all_anno_max = self.get_max_multilabels(self.y_all_anno_enc)

        if filter is not None:
            self.y_filt_anno_enc = self.update_labels(filter)
            self.y_filt_anno_max = self.get_max_multilabels(self.y_filt_anno_enc)
        else:
            self.y_filt_anno_enc = None
            self.y_filt_anno_max = None

        if self.plot_anno_flag:
            self.plot_annotations()

    def get_max_multilabels(self, y_in):
        y_out = np.zeros(y_in.shape)
        for i in range(y_in.shape[0]):
            if self.multi_label:
                # multi class
                y_out[i, np.argwhere(y_in[i, :] == np.amax(y_in[i, :]))] = 1
            else:
                # single class
                y_out[i, np.argmax(y_in[i, :])] = 1
        return y_out


    def plot_annotations(self):
        import matplotlib.gridspec as gridspec
        if self.group_rat:
            labels = self.list_quad
        else:
            labels = self.list_emo

        plot_all = np.zeros((len(labels) * 2, len(labels)))
        plot_filt = np.zeros((len(labels) * 2, len(labels)))
        plot_all_max = np.zeros((len(labels) * 2, len(labels)))
        plot_filt_max = np.zeros((len(labels) * 2, len(labels)))
        for j, i in enumerate(range(0, 528, 24)):
            plot_all[j, :] = self.y_all_anno_enc[i, :]
            plot_all_max[j, :] = self.y_all_anno_max[i, :]
            plot_filt[j, :] = self.y_filt_anno_enc[i, :]
            plot_filt_max[j, :] = self.y_filt_anno_max[i, :]

        labels[-1] = 'transc.'
        # plot ground truth
        color = '#ffffff'
        # color = '#f0f0f0'
        center = 0.5
        fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, constrained_layout=True)
        # ax[0, 0].imshow(self.y_oe, aspect='auto')
        # ax[0, 0].set_title('Orig. G.T.')
        im = ax[0, 0].imshow(plot_all, aspect='auto')
        ax[0, 0].set_ylabel('Excerpts')
        if self.group_rat is False:
            for i, j in zip(range(0, 12), range(0, 22, 2)):
                ax[0, 0].add_patch(patches.Rectangle((i-center, j-center), 1, 2, fill=False, edgecolor=color, linewidth=2))
        ax[0, 0].set_title('All ratings')
        ax[0, 1].imshow(plot_all_max, aspect='auto')
        if self.group_rat is False:
            for i, j in zip(range(0, 12), range(0, 22, 2)):
                ax[0, 1].add_patch(patches.Rectangle((i-center, j-center), 1, 2, fill=False, edgecolor=color, linewidth=2))
        ax[0, 1].set_title('All ratings (multi-label)')
        if self.y_filt_anno_enc is not None:
            # ax[1, 0].imshow(self.y_oe, aspect='auto')
            # ax[1, 0].set_xticks(np.arange(len(labels)))
            # ax[1, 0].set_xticklabels(labels, rotation=90)
            # ax[1, 0].set_title('Orig. G.T.')
            ax[1, 0].imshow(plot_filt, aspect='auto')
            ax[1, 0].set_ylabel('Excerpts')
            if self.group_rat is False:
                for i, j in zip(range(0, 12), range(0, 22, 2)):
                    ax[1, 0].add_patch(patches.Rectangle((i-center, j-center), 1, 2, fill=False, edgecolor=color, linewidth=2))
            ax[1, 0].set_xticks(np.arange(len(labels)))
            ax[1, 0].set_xticklabels(labels, rotation=90)
            ax[1, 0].set_title('Filt. ratings')
            ax[1, 1].imshow(plot_filt_max, aspect='auto')
            if self.group_rat is False:
                for i, j in zip(range(0, 12), range(0, 22, 2)):
                    ax[1, 1].add_patch(patches.Rectangle((i-center, j-center), 1, 2, fill=False, edgecolor=color, linewidth=2))
            ax[1, 1].set_xticks(np.arange(len(labels)))
            ax[1, 1].set_xticklabels(labels, rotation=90)
            ax[1, 1].set_title('Filt. ratings (multi-label)')
            
        # fig.tight_layout()
        fig.colorbar(im, ax=[ax[:, 1]], shrink=0.8)
        # plt.tight_layout()
        plt.show()
        # sys.exit(0)

        # get jaccard coefficients from annotations and ground truth
        jac_gt_all = jaccard_score(self.y_oe, self.y_all_anno_max, average='samples')
        jac_all_filt = jaccard_score(self.y_all_anno_max, self.y_filt_anno_max, average='samples')
        jac_gt_filt = jaccard_score(self.y_oe, self.y_filt_anno_max, average='samples')

        print('\n-------------------')
        print('Jaccard index GT vs all anno: {}'.format(jac_gt_all))
        print('Jaccard index GT vs filt anno: {}'.format(jac_gt_filt))
        print('Jaccard index all anno vs filt anno: {}'.format(jac_all_filt))
        print('-------------------\n')


    def get_all_feats(self):
        zmcsuv = True
        X = []
        y = []
        self.rep_list = []
        for f_f, anno in zip(self.list_feats_files, self.anno):
            this_f = pd.read_csv(f_f, sep=';').drop(columns=['frameTime'])
            X.append(this_f.values)
            y.append(np.repeat(anno, this_f.shape[0]).tolist())
            self.rep_list.append(this_f.shape[0])
        X = np.vstack(X)
        
        if zmcsuv:
            X = StandardScaler().fit_transform(X)
            self.C = 10
            self.gamma = 0.001
        else:
            X = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)
            self.C = 10
            self.gamma = 0.1

        # PCA 3 components - expl.variance: 0.43069215110107245
        if self.num_comp > 0:
            pca = PCA(n_components=self.num_comp, random_state=self.seed)
            X = pca.fit_transform(X)
            expl = np.sum(pca.explained_variance_ratio_[:self.num_comp + 1])
            print('------\nPCA {} components - expl.variance: {}\n------'.format(self.num_comp, expl))
            # plt.plot(range(20), pca.explained_variance_ratio_[:20])
            # plt.show()


        y = np.array([_ for x in y for _ in x])
        self.le = LabelEncoder()
        y_enc = self.le.fit_transform(y)

        self.oe = OneHotEncoder(sparse=False)
        y_oe = self.oe.fit_transform(y_enc[:, np.newaxis])

        print('\nClasses:\n', {i: _ for i, _ in enumerate(self.le.classes_)})
        return X, y_enc, y_oe


    def update_labels(self, filter):
        if filter is None:
            filt_txt = 'Not filtered data'
        elif filter == 'p1':
            self.ratings = self.ratings[self.ratings['pref_enc'] == '1']
            filt_txt = 'Positive preference (>3)'
        elif filter == 'p2':
            self.ratings = self.ratings[self.ratings['pref_enc'] == '0']
            filt_txt = 'Negative preference (<3)'
        elif filter == 'f1':
            self.ratings = self.ratings[self.ratings['fam_enc'] == '1']
            filt_txt = 'Positive familiarity (>3)'
        elif filter == 'f2':
            self.ratings = self.ratings[self.ratings['fam_enc'] == '0']
            filt_txt = 'Negative familiarity (<3)'
        elif filter == 'u1':
            self.ratings = self.ratings[self.ratings['und_enc'] == '1']
            filt_txt = 'Positive understanding (>3)'
        elif filter == 'u2':
            self.ratings = self.ratings[self.ratings['und_enc'] == '0']
            filt_txt = 'Negative understanding (<3)'
        elif filter == 'fm1':
            self.ratings = self.ratings[self.ratings['mus_train_encod'] == 1]
            filt_txt = 'Positive musical training (>mean)'
        elif filter == 'fm2':
            self.ratings = self.ratings[self.ratings['mus_train_encod'] == 0]
            filt_txt = 'Negative musical training (<mean)'
        elif filter == 'fe1':
            self.ratings = self.ratings[self.ratings['emotions_enc'] == 1]
            filt_txt = 'Positive emotion perception (>mean)'
        elif filter == 'fe2':
            self.ratings = self.ratings[self.ratings['emotions_enc'] == 0]
            filt_txt = 'Negative emotion perception (<mean)'
        elif filter == 'fs1':
            self.ratings = self.ratings[self.ratings['mus_soph_enc'] == 1]
            filt_txt = 'Positive music sophistication (>mean)'
        elif filter == 'fs2':
            self.ratings = self.ratings[self.ratings['mus_soph_enc'] == 0]
            filt_txt = 'Negative music sophistication (<mean)'
        else:
            print('Select a valid filter from [p1, p2, f1, f2, u1, u2, fm1, fm2, fe1, fe2, fs1, fs2]')
            sys.exit(0)
        self.filt_txt = 'Working with {} samples and {} ratings\n'.format(filt_txt, self.ratings.shape[0])
        list_files = [_.split('/')[-1].split('.')[0] for _ in self.list_feats_files]
        quad_enc = {'q1_a_pos_v_pos': ['joy', 'power', 'surprise'],
                    'q2_a_pos_v_neg': ['anger', 'fear', 'tension'],
                    'q3_a_neg_v_neg': ['bitter', 'sad'],
                    'q4_a_neg_v_pos': ['peace', 'tender', 'transcendence']}

        new_y = []
        
        if self.plot_hist:
            fig, ax = plt.subplots(nrows=len(self.list_songs), ncols=len(np.unique(self.anno)), figsize=(15, 30))
            cnt_norm = 0
        for i, song in enumerate(list_files):
            quad_rats = []
            if self.group_rat:
                tmp_rat = self.ratings[self.ratings.excerpt == song]
                for key, list_emo in quad_enc.items():
                    quad_rats.append(np.median(tmp_rat[list_emo].values, axis=1))   
                rat_this_song = np.vstack(quad_rats).T
            else:
                rat_this_song = np.array(self.ratings[self.ratings.excerpt == song].filter(self.list_emo))

            from scipy import stats

            if self.plot_hist:
                for j in range(rat_this_song.shape[1]):
                    ax[i, j].hist(rat_this_song[:, j])
                    try:
                        # k, p = stats.normaltest(rat_this_song[:, j])
                        # if p < 1e-3:
                        #     ax[i, j].text(1, 0, 'not normal')
                        #     cnt_norm += 1
                        pos_cnt = np.sum(rat_this_song[:, j] > 3)
                        neg_cnt = np.sum(rat_this_song[:, j] < 3)
                        # rat_this_song[:, j] = 3
                        if pos_cnt > neg_cnt:
                            rat_this_song[:, j] = 4
                        elif pos_cnt < neg_cnt:
                            rat_this_song[:, j] = 2
                        else:
                            rat_this_song[:, j] = 3
                    except:
                        print('Song {} - {} has less than 8 samples'.format(song, np.unique(self.anno)[j]))
                    if i == 0:
                        ax[i, j].set_title(np.unique(self.anno)[j])
                    if j == 0:
                        ax[i, j].set_ylabel(song)
                    ax[i, j].set_xlim([0, 6])

            # fuzz_lab = np.mean(rat_this_song, axis=0).tolist()
            # fuzz_lab = np.median(rat_this_song, axis=0).tolist()
            fuzz_lab = stats.mode(rat_this_song, axis=0)[0][0].tolist()
            # print(song)
            # print(fuzz_lab)
            # print(rat_this_song.shape)
            new_y.append(np.repeat([fuzz_lab], self.rep_list[i], axis=0))

        if self.plot_hist:
            plt.tight_layout()
            plt.savefig('histograms.png')
            plt.close(fig)
            print('\n{} from {} ratings per song could be normal (p<0.001): {}\n'.format(len(self.list_emo) * len(self.list_songs) - cnt_norm,
                                                                                 len(self.list_emo) * len(self.list_songs),
                                                                                 1 - cnt_norm/(len(self.list_emo) * len(self.list_songs))))
        y_new = np.vstack(new_y)

        return y_new


    def print_cluster_stats(self, labels, labels_filt, clust_labels, clust_labels_filt):
        from sklearn import metrics
        rand_all = metrics.adjusted_rand_score(labels, clust_labels)
        mutual_all = metrics.adjusted_mutual_info_score(labels, clust_labels, average_method='arithmetic')

        if labels_filt is not None and clust_labels_filt is not None:
            rand_filt = metrics.adjusted_rand_score(labels_filt, clust_labels_filt)
            mutual_filt = metrics.adjusted_mutual_info_score(labels_filt, clust_labels_filt, average_method='arithmetic')
        else:
            rand_filt = ''
            mutual_filt = ''
        print('**********************')
        print('Scores:')
        print('Rand score all data: {}, Rand score with filter: {}'.format(rand_all, rand_filt))
        print('Mutual info score all data: {}, Mutual info score with filter: {}'.format(mutual_all, mutual_filt))
        print('**********************')


    def run_data(self, flag, model_name):
        if flag == 'orig':
            print('\n(¯`·._.·(¯`·._.· Evaluation on original data ·._.·´¯)·._.·´¯)\n')
            y = self.y
            y_pca = y
        elif flag == 'full_anno':
            print('\n(¯`·._.·(¯`·._.· Evaluation on full annotations ·._.·´¯)·._.·´¯)\n')
            print(self.filt_txt)
            y = self.y_all_anno_max
            y_pca = np.ravel(self.oe.inverse_transform(y))
        elif flag == 'filt':
            print('\n(¯`·._.·(¯`·._.· Evaluation on filtered data ·._.·´¯)·._.·´¯)\n')
            print(self.filt_txt)
            y = self.y_filt_anno_max
            y_pca = np.ravel(self.oe.inverse_transform(y))
        if self.pca_plot:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.X[:, 0], self.X[:, 1], self.X[:, 2], c=y_pca)
            plt.show()

        if model_name == 'svm':
            # # support vector classifier
            if self.multi_label is False:
                if flag != 'orig':
                    y = np.ravel(self.oe.inverse_transform(y))
                kfold = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
                clf_cv = svm.SVC(C=self.C, gamma=self.gamma, random_state=self.seed)
            # if multilabel
            else:
                if flag != 'orig':
                    kfold = IterativeStratification(n_splits=5, order=1, random_state=self.seed)
                else:
                    # kfold = model_selection.KFold(n_splits=5, random_state=self.seed)
                    kfold = model_selection.StratifiedKFold(n_splits=5, random_state=self.seed)
            
                clf_cv = OneVsRestClassifier(svm.SVC(C=self.C, gamma=self.gamma, random_state=self.seed))
                # clf_cv = svm.SVC(C=self.C, gamma=self.gamma, random_state=1987)

            # best_params = self.svc_param_selection(clf_cv, self.X, self.y, kfold)
            # print('Best params:', best_params)
            pre = model_selection.cross_val_score(clf_cv, self.X, y, cv=kfold, scoring='precision_macro')
            rec = model_selection.cross_val_score(clf_cv, self.X, y, cv=kfold, scoring='recall_macro')
            fsc = model_selection.cross_val_score(clf_cv, self.X, y, cv=kfold, scoring='f1_macro')
            # score, perm_sc, pvalue = model_selection.permutation_test_score(clf_cv, self.X, self.y, cv=kfold, scoring='f1_macro', n_permutations=100, n_jobs=-1)

            print('5-Fold CV Precision: {} , STD: {}'.format(pre.mean(), pre.std()))
            print('5-Fold CV Recall: {} , STD: {}'.format(rec.mean(), rec.std()))
            print('5-Fold CV F1-Score: {} , STD: {}'.format(fsc.mean(), fsc.std()))

            # print('Classification score {} (pvalue : {})'.format(score, pvalue))
            # X_train, X_test, y_train, y_test = model_selection.train_test_split(self.X, y, test_size=0.5, random_state=1987)

            # clf_cv.fit(X_train, y_train)
            # y_pred = clf_cv.predict(X_test)

            # report = classification_report(y_test, y_pred)
            # print(report)

            return pre, rec, fsc

        elif model_name == 'gmm':
            # gaussian mixture model
            clf_cv = mixture.GaussianMixture(n_components=len(np.unique(y)), covariance_type='full', n_init=10, random_state=1987)
            # clf_cv = mixture.BayesianGaussianMixture(n_components=len(np.unique(self.y)), covariance_type='full', n_init=10, random_state=1987)
            clf_cv.fit(self.X)
            y_pred = clf_cv.predict(self.X)
            # pdb.set_trace()
            return y, y_pred


    def svc_param_selection(self, clf, X, y, nfolds):
        Cs = [0.001, 0.01, 0.1, 1, 10]
        gammas = [0.001, 0.01, 0.1, 1]
        param_grid = {'C': Cs, 'gamma': gammas}
        # param_grid = {'estimator__C': Cs, 'estimator__gamma': gammas}
        grid_search = model_selection.GridSearchCV(clf, param_grid, cv=nfolds)
        grid_search.fit(X, y)
        grid_search.best_params_
        return grid_search.best_params_


    def plot_res(self, pre_orig, rec_orig, fsc_orig, pre_full, rec_full, fsc_full, pre_filt, rec_filt, fsc_filt):
        labels = ['Prec.', 'Rec.', 'F1-Sc.']
        means_orig = [np.mean(pre_orig), np.mean(rec_orig), np.mean(fsc_orig)]
        std_orig = [np.std(pre_orig), np.std(rec_orig), np.std(fsc_orig)]
        means_full = [np.mean(pre_full), np.mean(rec_full), np.mean(fsc_full)]
        std_full = [np.std(pre_full), np.std(rec_full), np.std(fsc_full)]
        means_filt = [np.mean(pre_filt), np.mean(rec_filt), np.mean(fsc_filt)]
        std_filt = [np.std(pre_filt), np.std(rec_filt), np.std(fsc_filt)]
        impr1 = np.array(means_full) - np.array(means_orig)
        impr2 = np.array(means_filt) - np.array(means_full)
        impr3 = np.array(means_filt) - np.array(means_orig)
        width = 0.35

        plt.subplot(141)
        plt.bar(np.arange(len(labels)) - width/2, means_orig, width/3, yerr=std_orig, align='center', alpha=0.4, label='Orig', capsize=2)
        plt.bar(np.arange(len(labels)), means_full, width/3, yerr=std_full, align='center', alpha=0.4, label='Full', capsize=2)
        plt.bar(np.arange(len(labels)) + width/2, means_filt, width/3, yerr=std_filt, align='center', alpha=0.5, label='Filt', capsize=2)
        plt.xticks(np.arange(len(labels)), (labels))
        plt.legend(loc='lower right')
        plt.ylabel('Scores')
        plt.subplot(142)
        plt.bar(np.arange(len(labels)), impr1, align='center', alpha=0.5)
        plt.ylabel('Diff. Score [All - MD]')
        plt.xticks(np.arange(len(labels)), (labels))
        plt.subplot(143)
        plt.bar(np.arange(len(labels)), impr3, align='center', alpha=0.5)
        plt.ylabel('Diff. Score [filt - MD]')
        plt.xticks(np.arange(len(labels)), (labels))
        plt.subplot(144)
        plt.bar(np.arange(len(labels)), impr2, align='center', alpha=0.5)
        plt.ylabel('Diff. Score [filt - All]')
        plt.xticks(np.arange(len(labels)), (labels))
        plt.tight_layout()
        plt.show()
        print('\n(¯`·._.·(¯`·._.· Final Summary ·._.·´¯)·._.·´¯)\n')
        print('Improvement All - MD:\n     Precision: {}\n     Recall: {}\n     F-Score: {}\n'.format(impr1[0], impr1[1], impr1[2]))
        print('Improvement Filt - MD:\n     Precision: {}\n     Recall: {}\n     F-Score: {}\n'.format(impr3[0], impr3[1], impr3[2]))
        print('Improvement Filt - All:\n     Precision: {}\n     Recall: {}\n     F-Score: {}\n'.format(impr2[0], impr2[1], impr2[2]))
        


if __name__ == "__main__":
    warnings.filterwarnings(action='ignore')

    # usage python3 classifier.py -l [a/e/s/m/g] -q [y/n] -nc int -f [p1,p2,f1,f2,u1,u2]
    parser = argparse.ArgumentParser()
    parser.add_argument('-l',
                        '--language',
                        help='Select language of surveys to process: [e,g,s,m,a]',
                        required=True,
                        action='store')
    parser.add_argument('-q',
                        '--quadrant',
                        help='Process by quadrants [y] or by emotions [n]',
                        required=True,
                        action='store')
    parser.add_argument('-m',
                        '--model',
                        help='Select model [svm, gmm]',
                        required=True,
                        action='store')
    parser.add_argument('-nc',
                        '--num_comp',
                        help='Select number of components for PCA (0 is no PCA)',
                        required=True,
                        action='store')
    parser.add_argument('-f',
                        '--filter',
                        help='Select filter for data [none (none), preference (p1, p2), familiarity (f1, f2), understanding (u1, u2), music training (fm1, fm2), emotion perception (fe1, fe2), general sophistication (fs1, fs2)]',
                        required=True,
                        action='store')
    args = parser.parse_args()

    classifier = SVMClassifier(args.language, args.quadrant, args.filter, int(args.num_comp))

    if args.model == 'svm':
        pre_orig, rec_orig, fsc_orig = classifier.run_data('orig', args.model)

        pre_full, rec_full, fsc_full = classifier.run_data('full_anno', args.model)

        pre_filt, rec_filt, fsc_filt = classifier.run_data('filt', args.model)

        classifier.plot_res(pre_orig, rec_orig, fsc_orig, pre_full, rec_full, fsc_full, pre_filt, rec_filt, fsc_filt)

    elif args.model == 'gmm':
        labels, clust_labels = classifier.run_data('full', args.model)

        labels_filt, clust_labels_filt = classifier.run_data('filt', args.model)

        classifier.print_cluster_stats(labels, labels_filt, clust_labels, clust_labels_filt)
    # classifier.run_data_reg()
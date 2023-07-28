from sklearn.datasets import make_moons
import numpy as np
import random
import matplotlib.pyplot as plt
from da_ot import OTGroupLassoDAClassifier, OTBFBDAClassifier
import ot
from sklearn.neighbors import KNeighborsClassifier
from data import Xt_all, Xt_all_domain, load_battery_data_split
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def test_cdot_methods(methods, time_reg_vector, n_samples_source, n_samples_targets,
                      time_length, time_series, sort_method, if_sort, methods_names=None, cost="seq",
                      fig_name=None, plot_mapping=False, random_seed = 0, preg=False, gamma_path=None):
    
    # Xs, ys, Xt, yt, Xt_all, yt_all = load_battery_data(n_samples_source, n_samples_targets, time_length, True)
    # Xs, ys, Xt, yt, Xt_all, yt_all = load_battery_data_random(n_samples_source, n_samples_targets, time_series, shuffle_or_not = True, random_seed = random_seed)
    Xs, ys, Xt, yt, Xt_all, yt_all, acc, Xt_true, yt_true, Xt_random, yt_random, Xt_all_domain, yt_all_domain, Xt_all_domain_mix, yt_all_domain_mix = load_battery_data_split(n_samples_source, n_samples_targets, time_series, shuffle_or_not = True, random_seed = random_seed, train_set = 20)

    if sort_method == 'w_dis' or sort_method == 'mix':
        if sort_method == 'mix':
            Xt_all_domain = Xt_all_domain_mix
            yt_all_domain = yt_all_domain_mix

        m = ot.dist(Xs, Xt_true[-1], metric='euclidean')
        m /= m.max()
        n1 = Xs.shape[0]
        n2 = Xt_true[-1].shape[0]
        a, b = np.ones((n1,)) / n1, np.ones((n2,)) / n2
        c_t = ot.sinkhorn2(a, b, m, 1)

        w_dist = []
        for x in Xt_all_domain:
            m = ot.dist(Xs, x, metric='euclidean')
            m /= m.max()
            n1 = Xs.shape[0]
            n2 = x.shape[0]
            a, b = np.ones((n1,)) / n1, np.ones((n2,)) / n2
            c = ot.sinkhorn2(a, b, m, 1)
            w_dist.append(c)
        
        t = time_length - 1
        if if_sort == 1:

            w_dist_min = []
            Xt_min = []
            yt_min = []
            for i in range(len(w_dist)):
                if w_dist[i] <= c_t:
                    w_dist_min.append(w_dist[i])
                    Xt_min.append(Xt_all_domain[i])
                    yt_min.append(yt_all_domain[i])

            
            rand = np.arange(len(w_dist_min))
            np.random.seed(random_seed)
            np.random.shuffle(rand)

            Xt1 = [x for _, x in sorted(zip(rand, Xt_min), key=lambda x1: x1[0])]
            yt1 = [x for _, x in sorted(zip(rand, yt_min), key=lambda x1: x1[0])]
            w1 = [x for _, x in sorted(zip(rand, w_dist_min), key=lambda x1: x1[0])]
            
            if t <= len(Xt1):
                Xt1 = Xt1[:t]
                yt1 = yt1[:t]
                w1 = w1[:t]
                Xt = [x for _, x in sorted(zip(w1, Xt1), key=lambda x1: x1[0])]
                yt = [x for _, x in sorted(zip(w1, yt1), key=lambda x1: x1[0])]

            else:
                x_sample = []
                y_sample = []
                w_sample = []
                for i in range(t - len(Xt1)):
                    np.random.seed(1 * i)
                    rand = np.random.choice(len(Xt1), 1)
                    x_sample_per = np.array(Xt1[rand[0]])
                    y_sample_per = np.array(yt1[rand[0]])
                    w_sample_per = np.array(w1[rand[0]])
                    
                    x_sample.append(x_sample_per)
                    y_sample.append(y_sample_per)
                    w_sample.append(w_sample_per)

                Xt1 = Xt1 + x_sample
                yt1 = yt1 + y_sample
                w1 = w1 + w_sample
                
                Xt = [x for _, x in sorted(zip(w1, Xt1), key=lambda x1: x1[0])]
                yt = [x for _, x in sorted(zip(w1, yt1), key=lambda x1: x1[0])]

            Xt.append(Xt_true[-1])
            yt.append(yt_true[-1])

        if if_sort == 0:
            if sort_method == 'w_dis':
                Xt = Xt_true
                yt = yt_true
            else:
                rand = np.arange(len(Xt_all_domain))
                np.random.seed(random_seed)
                np.random.shuffle(rand)

                x_sample = []
                y_sample = []
                for i in range(t):
                    x_sample_per = np.array(Xt_all_domain[rand[i]])
                    y_sample_per = np.array(yt_all_domain[rand[i]])
                    
                    x_sample.append(x_sample_per)
                    y_sample.append(y_sample_per)

                Xt = x_sample
                yt = y_sample

                Xt.append(Xt_true[-1])
                yt.append(yt_true[-1])
            

    if sort_method == 'soc':
        Xt = Xt_true
        yt = yt_true
    
    if sort_method == 'random':
        Xt = Xt_random
        yt = yt_random
    
    mapped_samples = []
    time_reg = []
    entropic_reg = []

    scores = np.zeros([len(methods), time_length])
    losses = np.zeros([len(methods), time_length])
    ots = np.arange(time_length)

    for m, da_clf in enumerate(methods):
        # print("Running ", methods_names[m])
        temp_samples = []
        for k in range(time_length):
            if k > 0:
                if cost == "seq":
                    if preg:
                        da_clf.fit(temp_samples[-1], ys, Xt[k], treg=time_reg_vector[m], preg=50, Gamma_old=da_clf.Gamma,
                                Xt_old=Xt[k - 1], path_cons=True, G_path=gamma_path)
                    
                    else:
                        da_clf.fit(temp_samples[-1], ys, Xt[k], treg=time_reg_vector[m], Gamma_old=da_clf.Gamma,
                                Xt_old=Xt[k - 1])
                    
                    time_reg.append(da_clf.temp_reg(da_clf.Gamma))
                    # print("time_reg: ", time_reg)
                    entropic_reg.append(da_clf.entropic_reg(da_clf.Gamma))
                    # print("entropic reg: ", entropic_reg)
                    # print("--seq oting--")
                else:
                    # da_clf.fit(Xs, ys, Xt[k], treg=time_reg_vector[m], Gamma_old=da_clf.Gamma,
                    #            Xt_old=Xt[k - 1])
                    da_clf.fit(Xs, ys, Xt[k])
                    # print("--direct oting--")

            else:
                da_clf.fit(Xs, ys, Xt[k])

            temp_samples.append(da_clf.adapt_source_to_target())
            
            result = da_clf.predict(Xt_all[k])
            scores[m, k] = r2_score(yt_all[k], result)
            losses[m, k] = mean_squared_error(yt_all[k], result)

        mapped_samples.append(temp_samples)
        
    return scores, losses, ots, time_reg, entropic_reg, acc


per_epoch_loss_direct = []
per_epoch_loss_path1 = []
per_epoch_loss_path1_preg = []
per_epoch_loss_path2 = []

per_epoch_var_direct = []
per_epoch_var_path1 = []
per_epoch_var_path1_preg = []
per_epoch_var_path2 = []

path1 = []
path2 = []

for epoch in range(100):

    np.random.seed(epoch)
    ot_series = 5 * (np.random.choice(8, 2, replace=False) + 2)
    sorted_ot_series = np.sort(ot_series)
    sorted_ot_series = np.append(sorted_ot_series, [50])
    path1.append(sorted_ot_series)
    print("path1, ", path1)


    # direct
    clf = svm.SVR(gamma='scale')
    ot_BFB = OTBFBDAClassifier(clf, reg=0.2, regnorm=None, it=50, epochs=1000, lr=10, verbose=True)
    methods = [ot_BFB]
    methods_names =  ['ot_BFB']
    time_reg_vector = [50]

    cost=["direct"]
    final_scores = np.zeros([len(cost), 3])
    final_losses = np.zeros([len(cost), 3])

    final_scores_var = np.zeros([len(cost), 3])
    final_losses_var = np.zeros([len(cost), 3])
    for i, c in enumerate(cost):
        run_scores = []
        run_losses = []
        for run in range(10):
            # print("RUN %d..." % run)
            scores, losses, ots, time_reg, entropic_reg, acc = test_cdot_methods(
                methods=methods,
                methods_names=methods_names,
                time_reg_vector=time_reg_vector,
                fig_name="seq_run_" + str(run),
                time_length=3,
                time_series=sorted_ot_series,
                n_samples_source=67,
                n_samples_targets=10,
                plot_mapping=False,
                cost=c,
                random_seed = (epoch+1) * (run+1),
                sort_method = 'soc',
                if_sort=1
            )
            run_scores.append(scores)
            run_losses.append(losses)
        
        avg_scores = np.mean(np.array(run_scores), axis=0)
        score_var = np.var(np.array(run_scores), axis=0)
        
        avg_losses = np.mean(np.array(run_losses), axis=0)
        loss_var = np.var(np.array(run_losses), axis=0)
    
        final_scores[i, :] = avg_scores
        final_losses[i, :] = avg_losses

        final_scores_var[i, :] = score_var
        final_losses_var[i, :] = loss_var
    
    per_epoch_loss_direct.append(final_losses[0, -1])
    per_epoch_var_direct.append(final_losses_var[0, -1])
    print("direct loss, ", final_losses[0, -1])
    
    # path1
    clf = svm.SVR(gamma='scale')
    ot_BFB1 = OTBFBDAClassifier(clf, reg=0.2, regnorm=None, it=50, epochs=1000, lr=10, verbose=True)
    methods = [ot_BFB1]
    methods_names =  ['ot_BFB']
    time_reg_vector = [50]
    cost=["seq"]
    final_scores = np.zeros([len(cost), 3])
    final_losses = np.zeros([len(cost), 3])

    final_scores_var = np.zeros([len(cost), 3])
    final_losses_var = np.zeros([len(cost), 3])
    for i, c in enumerate(cost):
        run_scores = []
        run_losses = []
        for run in range(10):
            # print("RUN %d..." % run)
            scores, losses, ots, time_reg, entropic_reg, acc = test_cdot_methods(
                methods=methods,
                methods_names=methods_names,
                time_reg_vector=time_reg_vector,
                fig_name="seq_run_" + str(run),
                time_length=3,
                time_series=sorted_ot_series,
                n_samples_source=67,
                n_samples_targets=10,
                plot_mapping=False,
                cost=c,
                random_seed = (epoch+1) * (run+1),
                sort_method = 'soc',
                if_sort=1
            )
            run_scores.append(scores)
            run_losses.append(losses)
        
        avg_scores = np.mean(np.array(run_scores), axis=0)
        score_var = np.var(np.array(run_scores), axis=0)
        
        avg_losses = np.mean(np.array(run_losses), axis=0)
        loss_var = np.var(np.array(run_losses), axis=0)
    
        final_scores[i, :] = avg_scores
        final_losses[i, :] = avg_losses

        final_scores_var[i, :] = score_var
        final_losses_var[i, :] = loss_var
    
    per_epoch_loss_path1.append(final_losses[0, -1])
    per_epoch_var_path1.append(final_losses_var[0, -1])
    print("path1 loss, ", final_losses[0, -1])

    # path2
    np.random.seed((epoch+1)*2023)
    ot_series = 5 * (np.random.choice(8, 2, replace=False) + 2)
    sorted_ot_series = np.sort(ot_series)
    sorted_ot_series = np.append(sorted_ot_series, [50])
    path2.append(sorted_ot_series)
    print("path2, ", path2)
    
    clf = svm.SVR(gamma='scale')
    ot_BFB2 = OTBFBDAClassifier(clf, reg=0.2, regnorm=None, it=50, epochs=1000, lr=10, verbose=True)
    methods = [ot_BFB2]
    methods_names =  ['ot_BFB']
    time_reg_vector = [50]

    cost=["seq"]
    final_scores = np.zeros([len(cost), 3])
    final_losses = np.zeros([len(cost), 3])

    final_scores_var = np.zeros([len(cost), 3])
    final_losses_var = np.zeros([len(cost), 3])
    for i, c in enumerate(cost):
        run_scores = []
        run_losses = []
        for run in range(10):
            # print("RUN %d..." % run)
            scores, losses, ots, time_reg, entropic_reg, acc = test_cdot_methods(
                methods=methods,
                methods_names=methods_names,
                time_reg_vector=time_reg_vector,
                fig_name="seq_run_" + str(run),
                time_length=3,
                time_series=sorted_ot_series,
                n_samples_source=67,
                n_samples_targets=10,
                plot_mapping=False,
                cost=c,
                random_seed = (epoch+1) * (run+1),
                sort_method = 'soc',
                if_sort=1
            )
            run_scores.append(scores)
            run_losses.append(losses)
        
        avg_scores = np.mean(np.array(run_scores), axis=0)
        score_var = np.var(np.array(run_scores), axis=0)
        
        avg_losses = np.mean(np.array(run_losses), axis=0)
        loss_var = np.var(np.array(run_losses), axis=0)
    
        final_scores[i, :] = avg_scores
        final_losses[i, :] = avg_losses

        final_scores_var[i, :] = score_var
        final_losses_var[i, :] = loss_var
    
    per_epoch_loss_path2.append(final_losses[0, -1])
    per_epoch_var_path2.append(final_losses_var[0, -1])
    print("path2 loss, ", final_losses[0, -1])

    # refine path1
    np.random.seed(epoch)
    ot_series = 5 * (np.random.choice(8, 2, replace=False) + 2)
    sorted_ot_series = np.sort(ot_series)
    sorted_ot_series = np.append(sorted_ot_series, [50])
    print("refine path1, ", sorted_ot_series)
    
    methods = [ot_BFB1]
    methods_names =  ['ot_BFB']
    time_reg_vector = [50]
    cost=["seq"]
    final_scores = np.zeros([len(cost), 3])
    final_losses = np.zeros([len(cost), 3])

    final_scores_var = np.zeros([len(cost), 3])
    final_losses_var = np.zeros([len(cost), 3])
    for i, c in enumerate(cost):
        run_scores = []
        run_losses = []
        for run in range(10):
            # print("RUN %d..." % run)
            scores, losses, ots, time_reg, entropic_reg, acc = test_cdot_methods(
                methods=methods,
                methods_names=methods_names,
                time_reg_vector=time_reg_vector,
                fig_name="seq_run_" + str(run),
                time_length=3,
                time_series=sorted_ot_series,
                n_samples_source=67,
                n_samples_targets=10,
                plot_mapping=False,
                cost=c,
                random_seed = (epoch+1) * (run+1),
                sort_method = 'soc',
                if_sort=1,
                preg=True,
                gamma_path = ot_BFB2.Gamma
            )
            run_scores.append(scores)
            run_losses.append(losses)
        
        avg_scores = np.mean(np.array(run_scores), axis=0)
        score_var = np.var(np.array(run_scores), axis=0)
        
        avg_losses = np.mean(np.array(run_losses), axis=0)
        loss_var = np.var(np.array(run_losses), axis=0)
    
        final_scores[i, :] = avg_scores
        final_losses[i, :] = avg_losses

        final_scores_var[i, :] = score_var
        final_losses_var[i, :] = loss_var
    
    per_epoch_loss_path1_preg.append(final_losses[0, -1])
    per_epoch_var_path1_preg.append(final_losses_var[0, -1])
    print("refine path1 loss, ", final_losses[0, -1])


np.savez("path_consistency", path1=path1, path2=path2, path1_loss=per_epoch_loss_path1, 
         path2_loss=per_epoch_loss_path2, path1_refine_loss=per_epoch_loss_path1_preg, 
         direct_loss=per_epoch_loss_direct, path1_var=per_epoch_var_path1, 
         path2_var=per_epoch_var_path2,path1_refine_var=per_epoch_var_path1_preg,
         direct_var=per_epoch_var_direct)
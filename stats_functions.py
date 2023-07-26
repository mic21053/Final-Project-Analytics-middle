import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff
import scipy.stats as st
import random
import plotly.express as px
pd.options.mode.chained_assignment = None

def draw_correlation_graphs(df, target_feature_idx, columns_idxs, meth_list = ['spearman']):
    fig = make_subplots(rows = len(columns_idxs), cols = len(meth_list),
                        subplot_titles = tuple(map(str, range(1, len(columns_idxs) * len(meth_list) + 1))))
    for i, feature in enumerate([df.columns[i] for i in columns_idxs]):
        for j, meth in enumerate(meth_list):
            quant_low, quant_hi = 0, 1
            corr_list = []
            while quant_hi - quant_low > 0:
                q_low = df[feature].quantile(quant_low + 0.01)
                q_hi  = df[feature].quantile(quant_hi)
                df_filtered = df[(df[feature] < q_hi) & (df[feature] > q_low)]
                if len(df_filtered) == 0:
                    break
                diff1 = abs(df_filtered[feature].mean() - df_filtered[feature].median())
                if meth != 'pearson':
                    b = len(df_filtered) // 10
                    if b < 5:
                        break
                    df_filtered.loc[:, 'categorical'] = pd.cut(df_filtered[feature], bins=b, labels = False, include_lowest = True)
                    corr1 = df_filtered[[df.columns[target_feature_idx], 'categorical']].corr(method = meth).iloc[0, 1]
                    df_filtered = df_filtered.drop(columns = ['categorical'])
                else:
                    corr1 = df_filtered[[df.columns[target_feature_idx], feature]].corr(method=meth).iloc[0, 1]
                q_low = df[feature].quantile(quant_low)
                q_hi  = df[feature].quantile(quant_hi - 0.01)
                df_filtered = df[(df[feature] < q_hi) & (df[feature] > q_low)]
                if len(df_filtered) == 0:
                    break
                diff2 = abs(df_filtered[feature].mean() - df_filtered[feature].median())
                if meth != 'pearson':
                    b = len(df_filtered) // 10
                    if b < 5:
                        break
                    df_filtered.loc[:, 'categorical'] = pd.cut(df_filtered[feature], bins=b, labels = False, include_lowest = True)
                    corr2 = df_filtered[[df.columns[target_feature_idx], 'categorical']].corr(method = meth).iloc[0, 1]
                    df_filtered = df_filtered.drop(columns = ['categorical'])
                else:
                    corr2 = df_filtered[[df.columns[target_feature_idx], feature]].corr(method=meth).iloc[0, 1]
                if diff1 >= diff2:
                    quant_hi -= 0.01
                    corr_list.append([quant_low, quant_hi, corr2])
                else:
                    quant_low += 0.01
                    corr_list.append([quant_low, quant_hi, corr1])
            fig.add_trace(
                go.Scatter(y=pd.DataFrame(corr_list)[2]),
                row=i + 1, col=j + 1
                )
            vline_df = pd.DataFrame(corr_list)
            for p in [9, 19, 29, 39, 49]:
                fig.add_vline(x = p, line_width=1, line_dash='dash', line_color='red', row=i + 1, col=j + 1)
            fig.layout.annotations[j + i * len(meth_list)].update(text = f'Изменение корреляции<br>по методу {meth}<br>для параметра<br>{feature} ')
    fig.update_layout(height=len(columns_idxs) * 300, width=len(meth_list) * 400, showlegend=False)
    fig.show()

def bootstrep(df, target_feature_idx, columns_idxs, hypotheses_list, n = -1, N_TRIAL = 1000, func = np.mean, method = 'prop'):
    if len(columns_idxs) != len(hypotheses_list):
        print('ERROR: Количество исследуемых параметров и типов гипотез должно совпадать!')
        return
    if target_feature_idx == 1:
        threshold_list = []
        sub_index_list = []
        sub_index_list_2 = []
        data_list = []
        for rating_threshold in range(9, 4, -1):
            if method == 'by_func':
                threshold_list.extend([f'Оценки: Контрольная группа <= {rating_threshold}, Тестовая группа > {rating_threshold}' for _ in range(2)])
                sub_index_list.append('Разница контроль/тест по пересечению дов.интервалов')
                sub_index_list.append('Разница контроль/тест по вхождению нуля в дов.интервал')
            else:
                threshold_list.extend([f'Оценки: Контрольная группа <= {rating_threshold}, Тестовая группа > {rating_threshold}' for _ in range(10)])
                sub_index_list.extend([f'Разделение на контроль/тест по перцентилю {perc}' for perc in [0.9, 0.9, 0.8, 0.8, 0.7, 0.7, 0.6, 0.6, 0.5, 0.5]])
                sub_index_list_2.extend(['Разница контроль/тест по пересечению дов.интервалов' if i % 2 == 0 else 'Разница контроль/тест по значению p-value' for i in range(10)])
            test_df = df.loc[df.iloc[:, target_feature_idx].isin([rating for rating in range(10, rating_threshold, -1)]), :]
            control_df = df.loc[~df.iloc[:, target_feature_idx].isin([rating for rating in range(10, rating_threshold, -1)]), :]
            print('Разделим абонентов на тестовую и контрольную группы в зависимости от поставленной оценки:')
            control_list = [i for i in range(1, rating_threshold + 1)]
            test_list = [i for i in range(rating_threshold + 1, 11)]
            print(f'Контрольная группа - оценки {control_list}, тестовая группа - оценки {test_list}')
            print(100 * '-')
            print(f'Число абонентов контрольной группы - {len(control_df)}.')
            print(f'Число абонентов тестовой группы - {len(test_df)}.')
            data_list_1, data_list_2, data_list_3, data_list_4, data_list_5, data_list_6, data_list_7, data_list_8, data_list_9, data_list_10 = [], [], [], [], [], [], [], [], [], []
            for i, feature in enumerate([df.columns[i] for i in columns_idxs]):
                print(f'Проверим есть ли статистическая разница между группами по параметру {feature}')
                print('Нулевая гипотеза - "Разницы нет"')
                if hypotheses_list[i] == 1:
                    print('Альтернативная гипотеза - "Значение параметра у тестовой группы выше"')
                else:
                    print('Альтернативная гипотеза - "Значение параметра у тестовой группы ниже"')
                x = control_df[feature]
                y = test_df[feature]
                print(f'Сравним на графике распределения параметра {feature} для тестовой и контрольной группы')
                fig = ff.create_distplot([x, y], ['control', 'test'], bin_size=50, show_rug=False)
                fig.update_layout(width=800, height=600, bargap=0.01, title = f'Распределения {feature} для тестовой и контрольной группы')
                fig.show()
                if method == 'by_func':
                    print(f'Выбранный метод - по функции.')
                    res = []
                    if n == -1:
                        n_control = len(x)
                        n_test = len(y)
                    else:
                        n_control = n
                        n_test = n
                    for _ in range(N_TRIAL):
                        subsample_control = np.random.choice(x, size = (n_control,))
                        subsample_test = np.random.choice(y, size = (n_test,))
                        stat_control = func(subsample_control)
                        stat_test = func(subsample_test)
                        res.append([stat_control, stat_test])
                    control_test_df = pd.DataFrame(res, columns = ['Control', 'Test'])
                    fig = px.histogram(control_test_df,
                        x = ['Control', 'Test'],
                        title = 'Распределение функции для контрольной и тестовой групп',
                        barmode='overlay')
                    fig.update_traces(marker_line_width=1,marker_line_color="white")
                    left_control_perc = np.percentile(control_test_df['Control'],2.5)
                    right_control_perc = np.percentile(control_test_df['Control'],97.5)
                    left_test_perc = np.percentile(control_test_df['Test'],2.5)
                    right_test_perc = np.percentile(control_test_df['Test'],97.5)
                    fig.add_vline(x=left_control_perc,
                     line_width=1, line_dash='solid', line_color='red')
                    fig.add_vline(x=right_control_perc,
                     line_width=1, line_dash='solid', line_color='red')
                    fig.add_vline(x=left_test_perc,
                     line_width=1, line_dash='dash', line_color='green')
                    fig.add_vline(x=right_test_perc,
                     line_width=1, line_dash='dash', line_color='green')
                    fig.show()
                    if hypotheses_list[i] == 1:
                        if left_test_perc >= right_control_perc:
                            print('Доверительные интервалы не пересекаются.\nТестовая выборка на уровне доверия 95% лучше контрольной')
                            data_list_1.append('Да')
                        elif right_test_perc <= left_control_perc:
                            print('Доверительные интервалы не пересекаются.\nНо тестовая выборка на уровне доверия 95% хуже контрольной')
                            data_list_1.append('Тест хуже!')
                        else:
                            print('Доверительные интервалы пересекаются.\nНа уровне доверия 95% нельзя отвергнуть нулевую гипотезу.\nРазницы между тестовой и контрольной выборкой нет.')
                            data_list_1.append('Нет')
                    else:
                        if right_test_perc <= left_control_perc:
                            print('Доверительные интервалы не пересекаются.\nТестовая выборка на уровне доверия 95% лучше контрольной')
                            data_list_1.append('Да')
                        elif left_test_perc >= right_control_perc:
                            print('Доверительные интервалы не пересекаются.\nНо тестовая выборка на уровне доверия 95% хуже контрольной')
                            data_list_1.append('Тест хуже!')
                        else:
                            print('Доверительные интервалы пересекаются.\nНа уровне доверия 95% нельзя отвергнуть нулевую гипотезу.\nРазницы между тестовой и контрольной выборкой нет.')
                            data_list_1.append('Нет')
                    differences = []
                    for _ in range(N_TRIAL):
                        control_sample =  x.sample(n_control, replace = True)
                        test_sample =  y.sample(n_test, replace = True)
                        if left_test_perc >= right_control_perc or (left_test_perc < right_control_perc and right_test_perc > right_control_perc):
                            differences.append(func(test_sample)-func(control_sample))
                        elif right_test_perc <= left_control_perc or (right_test_perc > left_control_perc and left_test_perc < left_control_perc):
                            differences.append(func(control_sample)-func(test_sample))
                        else:
                            differences.append(func(test_sample)-func(control_sample))
                    differences_df = pd.DataFrame(differences, columns = ['Differences'])
                    fig = px.histogram(differences_df,
                        x = ['Differences'],
                        title = 'Распределение разниц средних контрольной и тестовой групп')
                    fig.update_traces(marker_line_width=1,marker_line_color="white")
                    left_diff_perc = np.percentile(differences_df['Differences'],2.5)
                    right_diff_perc = np.percentile(differences_df['Differences'],97.5)
                    fig.add_vline(x=left_diff_perc,
                     line_width=1, line_dash='solid', line_color='red')
                    fig.add_vline(x=right_diff_perc,
                     line_width=1, line_dash='solid', line_color='red')
                    fig.show()
                    if left_diff_perc <= 0 and right_diff_perc >= 0:
                        print('Ноль входит в доверительный интервал.\nНа уровне доверия 95% нельзя отвергнуть нулевую гипотезу.\nРазницы между тестовой и контрольной выборкой нет.')
                        data_list_2.append('Нет')
                    else:
                        print('Ноль не входит в доверительный интервал.\nТестовая выборка на уровне доверия 95% отлична от контрольной')
                        data_list_2.append('Да')
                    print(100 * '-')
                else:
                    print('Выберем некий порог параметра связи и для контрольной и тестовой групп посчитаем количество абонентов с параметром лучше порога и хуже порога.\nДальнейшие рассчеты будем вести на основании доли абонентов с хорошим параметром в каждой из групп.')
                    sub_data_list_1, sub_data_list_2 = [], []
                    for perc in [0.9, 0.8, 0.7, 0.6, 0.5]:
                        if hypotheses_list[i] == 1:
                            good_threshold = df[feature].quantile(perc)
                            # good_threshold = min(x.quantile(perc), y.quantile(perc))
                            control_df['good/bad param'] = control_df[feature].apply(lambda x: 1 if x >= good_threshold else 0)
                            test_df['good/bad param'] = test_df[feature].apply(lambda x: 1 if x >= good_threshold else 0)
                        else:
                            good_threshold = df[feature].quantile(1 - perc)
                            # good_threshold = max(x.quantile(1 - perc), y.quantile(1 - perc))
                            control_df['good/bad param'] = control_df[feature].apply(lambda x: 1 if x <= good_threshold else 0)
                            test_df['good/bad param'] = test_df[feature].apply(lambda x: 1 if x <= good_threshold else 0)
                        if hypotheses_list[i] == 1:
                            print(f'Разделим параметр {feature} на хороший и плохой по квантилю {perc}')
                        else:
                            print(f'Разделим параметр {feature} на хороший и плохой по квантилю {round(1 - perc, 1)}')
                        p_old = func(control_df['good/bad param'])
                        p_new = func(test_df['good/bad param'])
                        print(f'Тогда доля абонентов с хорошим параметром {feature}:')
                        print(f'в контрольной группе - {p_old}')
                        print(f'в тестовой группе - {p_new}')
                        print('Нулевая гипотеза - доли равны.')
                        print('Альтернативная гипотеза - доля абонентов с хорошим параметром в тестовой группе больше чем в контрольной.')
                        diff_base = p_new - p_old
                        print(f'Базовая разница долей - {diff_base}')
                        se_control = np.sqrt(p_old * (1 - p_old) / len(control_df))
                        left_control, right_control = st.norm.interval(0.95, p_old, se_control)
                        print('Доверительный интервал для доли абонентов с хорошим параметром в контрольной группе:')
                        print(f'от {left_control} до {right_control}.')
                        se_test = np.sqrt(p_new * (1 - p_new) / len(test_df))
                        left_test, right_test = st.norm.interval(0.95, p_new, se_test)
                        print('Доверительный интервал для доли абонентов с хорошим параметром в тестовой группе:')
                        print(f'от {left_test} до {right_test}.')
                        if left_test > right_control:
                            print('Доверительные интервалы не пересекаются - на уровне значимости 5% можно утверждать,\nчто статистическая разница между контрольной и тестовой группой есть.')
                            sub_data_list_1.append('Да')
                        else:
                            print('Доверительные интервалы пересекаются - на уровне значимости 5% нельзя утверждать,\nчто статистическая разница между контрольной и тестовой группой есть.')
                            sub_data_list_1.append('Нет')
                        differences = np.zeros((1, N_TRIAL))
                        for j in range(0, N_TRIAL):
                            if n == -1:
                                n_control = len(x)
                                n_test = len(y)
                            else:
                                n_control = n
                                n_test = n
                            s1 = random.choices(control_df['good/bad param'].to_numpy(), k = n_control)
                            s2 = random.choices(test_df['good/bad param'].to_numpy(), k = n_test)
                            p1 = func(s1)
                            p2 = func(s2)
                            differences[0][j] = p2 - p1
                        differences_cent = differences - np.mean(differences)
                        print('Посмотрим на график распределения разниц контрольной и тестовой выборок соответственно заданной функции среднего или медианы.')
                        fig = px.histogram(x = differences_cent[0], title = 'Распределение разниц контрольной и тестовой выборок')
                        fig.update_traces(marker_line_width=1,marker_line_color="white")
                        fig.add_vline(x=diff_base, line_width=1, line_dash='solid', line_color='red')
                        fig.show()
                        print('По графику p-value - это доля значений распределения правее красной линии.')
                        p_value = np.sum(differences_cent > diff_base) / N_TRIAL
                        print(f'Мы получили p-value = {p_value}.')
                        if p_value < 0.05:
                            print('P-value < 0.05, значит есть основания утверждать, что статистическая разница между контрольной и тестовой группой есть.')
                            sub_data_list_2.append('Да')
                        else:
                            print('P-value > 0.05, значит есть основания утверждать, что статистической разницы между контрольной и тестовой группой нет.')
                            sub_data_list_2.append('Нет')
                        print(100 * '-')
                    data_list_1.append(sub_data_list_1[0])
                    data_list_2.append(sub_data_list_2[0])
                    data_list_3.append(sub_data_list_1[1])
                    data_list_4.append(sub_data_list_2[1])
                    data_list_5.append(sub_data_list_1[2])
                    data_list_6.append(sub_data_list_2[2])
                    data_list_7.append(sub_data_list_1[3])
                    data_list_8.append(sub_data_list_2[3])
                    data_list_9.append(sub_data_list_1[4])
                    data_list_10.append(sub_data_list_2[4])
            if method == 'by_func':
                data_list.extend([data_list_1, data_list_2])
            else:
                data_list.extend([data_list_1, data_list_2, data_list_3, data_list_4,
                 data_list_5, data_list_6, data_list_7, data_list_8, data_list_9, data_list_10])
        feature_list = [df.columns[i] for i in columns_idxs]
        feature_list = [tuple([name]) for name in feature_list]
        if method == 'by_func':
            index_list = [elem for elem in zip(threshold_list, sub_index_list)]
            names_list = ['', '']
        else:
            index_list = [elem for elem in zip(threshold_list, sub_index_list, sub_index_list_2)]
            names_list = ['', '', '']
        data = {'index': index_list,
                'columns': feature_list,
                'data': data_list,
                'index_names': names_list,
                'column_names': ['']
        }
        return pd.DataFrame.from_dict(data, orient='tight')
    else:
        answers_dict = {1: 'Недозвоны, обрывы при звонках',
                2: 'Время ожидания гудков при звонке',
                3: 'Плохое качество связи в зданиях',
                4: 'Медленный мобильный Интернет',
                5: 'Медленная загрузка видео',
                6: 'Затрудняюсь ответить',
                7: 'Свой вариант'}
        threshold_list = []
        sub_index_list = []
        sub_index_list_2 = []
        data_list = []
        for answer in range(1, 7):
            if method == 'by_func':
                if answer == 6:
                    threshold_list.extend([f'Ответ контрольной группы "{answers_dict[6]}" и "{answers_dict[7]}", тестовая группа - все остальные' for _ in range(2)])
                else:
                    threshold_list.extend([f'Ответ контрольной группы "{answers_dict[answer]}", тестовая группа - все остальные' for _ in range(2)])
                sub_index_list.append('Разница контроль/тест по пересечению дов.интервалов')
                sub_index_list.append('Разница контроль/тест по вхождению нуля в дов.интервал')
            else:
                if answer == 6:
                    threshold_list.extend([f'Ответ контрольной группы "{answers_dict[6]}" и "{answers_dict[7]}", тестовая группа - все остальные' for _ in range(10)])
                else:
                    threshold_list.extend([f'Ответ контрольной группы "{answers_dict[answer]}", тестовая группа - все остальные' for _ in range(10)])
                sub_index_list.extend([f'Разделение на контроль/тест по перцентилю {perc}' for perc in [0.9, 0.9, 0.8, 0.8, 0.7, 0.7, 0.6, 0.6, 0.5, 0.5]])
                sub_index_list_2.extend(['Разница контроль/тест по пересечению дов.интервалов' if i % 2 == 0 else 'Разница контроль/тест по значению p-value' for i in range(10)])
            if answer == 6:
                control_df = df.loc[(df.iloc[:, target_feature_idx] == 6) | (df.iloc[:, target_feature_idx] == 7), :]
                test_df = df.loc[~df['user_id'].isin(control_df['user_id'])]
            else:
                control_df = df.loc[(df.iloc[:, target_feature_idx] == answer), :]
                test_df = df.loc[~df['user_id'].isin(control_df['user_id'])]
            print('Разделим абонентов на тестовую и контрольную группы в зависимости от выбранного ответа:')
            if answer == 6:
                print(f'Контрольная группа - давшие ответ "{answers_dict[6]}" и "{answers_dict[7]}", тестовая группа - все остальные')
            else:
                print(f'Контрольная группа - давшие ответ "{answers_dict[answer]}", тестовая группа - все остальные')
            print(100 * '-')
            print(f'Число абонентов контрольной группы - {len(control_df)}.')
            print(f'Число абонентов тестовой группы - {len(test_df)}.')
            data_list_1, data_list_2, data_list_3, data_list_4, data_list_5, data_list_6, data_list_7, data_list_8, data_list_9, data_list_10 = [], [], [], [], [], [], [], [], [], []
            for i, feature in enumerate([df.columns[i] for i in columns_idxs]):
                print(f'Проверим есть ли статистическая разница между группами по параметру {feature}')
                print('Нулевая гипотеза - "Разницы нет"')
                if hypotheses_list[i] == 1:
                    print('Альтернативная гипотеза - "Значение параметра у тестовой группы выше"')
                else:
                    print('Альтернативная гипотеза - "Значение параметра у тестовой группы ниже"')
                x = control_df[feature]
                y = test_df[feature]
                print(f'Сравним на графике распределения параметра {feature} для тестовой и контрольной группы')
                fig = ff.create_distplot([x, y], ['control', 'test'], bin_size=50, show_rug=False)
                fig.update_layout(width=800, height=600, bargap=0.01, title = f'Распределения {feature} для тестовой и контрольной группы')
                fig.show()
                if method == 'by_func':
                    print(f'Выбранный метод - по функции.')
                    res = []
                    if n == -1:
                        n_control = len(x)
                        n_test = len(y)
                    else:
                        n_control = n
                        n_test = n
                    for _ in range(N_TRIAL):
                        subsample_control = np.random.choice(x, size = (n_control,))
                        subsample_test = np.random.choice(y, size = (n_test,))
                        stat_control = func(subsample_control)
                        stat_test = func(subsample_test)
                        res.append([stat_control, stat_test])
                    control_test_df = pd.DataFrame(res, columns = ['Control', 'Test'])
                    fig = px.histogram(control_test_df,
                        x = ['Control', 'Test'],
                        title = 'Распределение функции для контрольной и тестовой групп',
                        barmode='overlay')
                    fig.update_traces(marker_line_width=1,marker_line_color="white")
                    left_control_perc = np.percentile(control_test_df['Control'],2.5)
                    right_control_perc = np.percentile(control_test_df['Control'],97.5)
                    left_test_perc = np.percentile(control_test_df['Test'],2.5)
                    right_test_perc = np.percentile(control_test_df['Test'],97.5)
                    fig.add_vline(x=left_control_perc,
                     line_width=1, line_dash='solid', line_color='red')
                    fig.add_vline(x=right_control_perc,
                     line_width=1, line_dash='solid', line_color='red')
                    fig.add_vline(x=left_test_perc,
                     line_width=1, line_dash='dash', line_color='green')
                    fig.add_vline(x=right_test_perc,
                     line_width=1, line_dash='dash', line_color='green')
                    fig.show()
                    if hypotheses_list[i] == 1:
                        if left_test_perc >= right_control_perc:
                            print('Доверительные интервалы не пересекаются.\nТестовая выборка на уровне доверия 95% лучше контрольной')
                            data_list_1.append('Да')
                        elif right_test_perc <= left_control_perc:
                            print('Доверительные интервалы не пересекаются.\nНо тестовая выборка на уровне доверия 95% хуже контрольной')
                            data_list_1.append('Тест хуже!')
                        else:
                            print('Доверительные интервалы пересекаются.\nНа уровне доверия 95% нельзя отвергнуть нулевую гипотезу.\nРазницы между тестовой и контрольной выборкой нет.')
                            data_list_1.append('Нет')
                    else:
                        if right_test_perc <= left_control_perc:
                            print('Доверительные интервалы не пересекаются.\nТестовая выборка на уровне доверия 95% лучше контрольной')
                            data_list_1.append('Да')
                        elif left_test_perc >= right_control_perc:
                            print('Доверительные интервалы не пересекаются.\nНо тестовая выборка на уровне доверия 95% хуже контрольной')
                            data_list_1.append('Тест хуже!')
                        else:
                            print('Доверительные интервалы пересекаются.\nНа уровне доверия 95% нельзя отвергнуть нулевую гипотезу.\nРазницы между тестовой и контрольной выборкой нет.')
                            data_list_1.append('Нет')
                    differences = []
                    for _ in range(N_TRIAL):
                        control_sample =  x.sample(n_control, replace = True)
                        test_sample =  y.sample(n_test, replace = True)
                        if left_test_perc >= right_control_perc or (left_test_perc < right_control_perc and right_test_perc > right_control_perc):
                            differences.append(func(test_sample)-func(control_sample))
                        elif right_test_perc <= left_control_perc or (right_test_perc > left_control_perc and left_test_perc < left_control_perc):
                            differences.append(func(control_sample)-func(test_sample))
                        else:
                            differences.append(func(test_sample)-func(control_sample))
                    differences_df = pd.DataFrame(differences, columns = ['Differences'])
                    fig = px.histogram(differences_df,
                        x = ['Differences'],
                        title = 'Распределение разниц средних контрольной и тестовой групп')
                    fig.update_traces(marker_line_width=1,marker_line_color="white")
                    left_diff_perc = np.percentile(differences_df['Differences'],2.5)
                    right_diff_perc = np.percentile(differences_df['Differences'],97.5)
                    fig.add_vline(x=left_diff_perc,
                     line_width=1, line_dash='solid', line_color='red')
                    fig.add_vline(x=right_diff_perc,
                     line_width=1, line_dash='solid', line_color='red')
                    fig.show()
                    if left_diff_perc <= 0 and right_diff_perc >= 0:
                        print('Ноль входит в доверительный интервал.\nНа уровне доверия 95% нельзя отвергнуть нулевую гипотезу.\nРазницы между тестовой и контрольной выборкой нет.')
                        data_list_2.append('Нет')
                    else:
                        print('Ноль не входит в доверительный интервал.\nТестовая выборка на уровне доверия 95% отлична от контрольной')
                        data_list_2.append('Да')
                    print(100 * '-')
                else:
                    print('Выберем некий порог параметра связи и для контрольной и тестовой групп посчитаем количество абонентов с параметром лучше порога и хуже порога.\nДальнейшие рассчеты будем вести на основании доли абонентов с хорошим параметром в каждой из групп.')
                    sub_data_list_1, sub_data_list_2 = [], []
                    for perc in [0.9, 0.8, 0.7, 0.6, 0.5]:
                        if hypotheses_list[i] == 1:
                            good_threshold = df[feature].quantile(perc)
                            # good_threshold = min(x.quantile(perc), y.quantile(perc))
                            control_df['good/bad param'] = control_df[feature].apply(lambda x: 1 if x >= good_threshold else 0)
                            test_df['good/bad param'] = test_df[feature].apply(lambda x: 1 if x >= good_threshold else 0)
                        else:
                            good_threshold = df[feature].quantile(1 - perc)
                            # good_threshold = max(x.quantile(1 - perc), y.quantile(1 - perc))
                            control_df['good/bad param'] = control_df[feature].apply(lambda x: 1 if x <= good_threshold else 0)
                            test_df['good/bad param'] = test_df[feature].apply(lambda x: 1 if x <= good_threshold else 0)
                        if hypotheses_list[i] == 1:
                            print(f'Разделим параметр {feature} на хороший и плохой по квантилю {perc}')
                        else:
                            print(f'Разделим параметр {feature} на хороший и плохой по квантилю {round(1 - perc, 1)}')
                        p_old = func(control_df['good/bad param'])
                        p_new = func(test_df['good/bad param'])
                        print(f'Тогда доля абонентов с хорошим параметром {feature}:')
                        print(f'в контрольной группе - {p_old}')
                        print(f'в тестовой группе - {p_new}')
                        print('Нулевая гипотеза - доли равны.')
                        print('Альтернативная гипотеза - доля абонентов с хорошим параметром в тестовой группе больше чем в контрольной.')
                        diff_base = p_new - p_old
                        print(f'Базовая разница долей - {diff_base}')
                        se_control = np.sqrt(p_old * (1 - p_old) / len(control_df))
                        left_control, right_control = st.norm.interval(0.95, p_old, se_control)
                        print('Доверительный интервал для доли абонентов с хорошим параметром в контрольной группе:')
                        print(f'от {left_control} до {right_control}.')
                        se_test = np.sqrt(p_new * (1 - p_new) / len(test_df))
                        left_test, right_test = st.norm.interval(0.95, p_new, se_test)
                        print('Доверительный интервал для доли абонентов с хорошим параметром в тестовой группе:')
                        print(f'от {left_test} до {right_test}.')
                        if left_test > right_control:
                            print('Доверительные интервалы не пересекаются - на уровне значимости 5% можно утверждать,\nчто статистическая разница между контрольной и тестовой группой есть.')
                            sub_data_list_1.append('Да')
                        else:
                            print('Доверительные интервалы пересекаются - на уровне значимости 5% нельзя утверждать,\nчто статистическая разница между контрольной и тестовой группой есть.')
                            sub_data_list_1.append('Нет')
                        differences = np.zeros((1, N_TRIAL))
                        for j in range(0, N_TRIAL):
                            if n == -1:
                                n_control = len(x)
                                n_test = len(y)
                            else:
                                n_control = n
                                n_test = n
                            s1 = random.choices(control_df['good/bad param'].to_numpy(), k = n_control)
                            s2 = random.choices(test_df['good/bad param'].to_numpy(), k = n_test)
                            p1 = func(s1)
                            p2 = func(s2)
                            differences[0][j] = p2 - p1
                        differences_cent = differences - np.mean(differences)
                        print('Посмотрим на график распределения разниц контрольной и тестовой выборок соответственно заданной функции среднего или медианы.')
                        fig = px.histogram(x = differences_cent[0], title = 'Распределение разниц контрольной и тестовой выборок')
                        fig.update_traces(marker_line_width=1,marker_line_color="white")
                        fig.add_vline(x=diff_base, line_width=1, line_dash='solid', line_color='red')
                        fig.show()
                        print('По графику p-value - это доля значений распределения правее красной линии.')
                        p_value = np.sum(differences_cent > diff_base) / N_TRIAL
                        print(f'Мы получили p-value = {p_value}.')
                        if p_value < 0.05:
                            print('P-value < 0.05, значит есть основания утверждать, что статистическая разница между контрольной и тестовой группой есть.')
                            sub_data_list_2.append('Да')
                        elif p_value > 0.95:
                            print('P-value > 0.95, значит есть основания утверждать, что статистическая разница между контрольной и тестовой группой есть, но верна гипотеза, обратная альтернативной.')
                            sub_data_list_2.append('Да, обрат.гип.')
                        else:
                            print('P-value > 0.05, значит есть основания утверждать, что статистической разницы между контрольной и тестовой группой нет.')
                            sub_data_list_2.append('Нет')
                        print(100 * '-')
                    data_list_1.append(sub_data_list_1[0])
                    data_list_2.append(sub_data_list_2[0])
                    data_list_3.append(sub_data_list_1[1])
                    data_list_4.append(sub_data_list_2[1])
                    data_list_5.append(sub_data_list_1[2])
                    data_list_6.append(sub_data_list_2[2])
                    data_list_7.append(sub_data_list_1[3])
                    data_list_8.append(sub_data_list_2[3])
                    data_list_9.append(sub_data_list_1[4])
                    data_list_10.append(sub_data_list_2[4])
            if method == 'by_func':
                data_list.extend([data_list_1, data_list_2])
            else:
                data_list.extend([data_list_1, data_list_2, data_list_3, data_list_4,
                 data_list_5, data_list_6, data_list_7, data_list_8, data_list_9, data_list_10])
        feature_list = [df.columns[i] for i in columns_idxs]
        feature_list = [tuple([name]) for name in feature_list]
        if method == 'by_func':
            index_list = [elem for elem in zip(threshold_list, sub_index_list)]
            names_list = ['', '']
        else:
            index_list = [elem for elem in zip(threshold_list, sub_index_list, sub_index_list_2)]
            names_list = ['', '', '']
        data = {'index': index_list,
                'columns': feature_list,
                'data': data_list,
                'index_names': names_list,
                'column_names': ['']
        }
        return pd.DataFrame.from_dict(data, orient='tight')

def p_value_graph(df, target_feature_idx, column_idx, hypotheses_type, n = -1, N_TRIAL = 1000, func = np.mean, alternative_hyp = 'more'):
    if target_feature_idx == 1:
        fig = make_subplots(rows = 5, cols = 1, subplot_titles = ('1', '2', '3', '4', '5'))
        for i, rating_threshold in enumerate(range(9, 4, -1)):
            test_df = df.loc[df.iloc[:, target_feature_idx].isin([rating for rating in range(10, rating_threshold, -1)]), :]
            control_df = df.loc[~df.iloc[:, target_feature_idx].isin([rating for rating in range(10, rating_threshold, -1)]), :]
            feature = df.columns[column_idx]
            x = control_df[feature]
            y = test_df[feature]
            p_value_list, perc_list = [], []
            for perc in np.linspace(0.95, 0.05, 19):
                perc = round(perc, 2)
                if hypotheses_type == 1:
                    good_threshold = df[feature].quantile(perc)
                    # good_threshold = min(x.quantile(perc), y.quantile(perc))
                    control_df['good/bad param'] = control_df[feature].apply(lambda x: 1 if x >= good_threshold else 0)
                    test_df['good/bad param'] = test_df[feature].apply(lambda x: 1 if x >= good_threshold else 0)
                    perc_list.append(perc)
                else:
                    good_threshold = df[feature].quantile(1 - perc)
                    # good_threshold = max(x.quantile(1 - perc), y.quantile(1 - perc))
                    control_df['good/bad param'] = control_df[feature].apply(lambda x: 1 if x <= good_threshold else 0)
                    test_df['good/bad param'] = test_df[feature].apply(lambda x: 1 if x <= good_threshold else 0)
                    perc_list.append(1 - perc)
                p_old = func(control_df['good/bad param'])
                p_new = func(test_df['good/bad param'])
                diff_base = p_new - p_old
                differences = np.zeros((1, N_TRIAL))
                for j in range(0, N_TRIAL):
                    if n == -1:
                        n_control = len(x)
                        n_test = len(y)
                    else:
                        n_control = n
                        n_test = n
                    s1 = random.choices(control_df['good/bad param'].to_numpy(), k = n_control)
                    s2 = random.choices(test_df['good/bad param'].to_numpy(), k = n_test)
                    p1 = func(s1)
                    p2 = func(s2)
                    differences[0][j] = p2 - p1
                differences_cent = differences - np.mean(differences)
                if alternative_hyp == 'more':
                    p_value = np.sum(differences_cent > diff_base) / N_TRIAL
                else:
                    p_value = np.sum(differences_cent < diff_base) / N_TRIAL
                p_value_list.append(p_value)
            fig.add_trace(
                go.Scatter(x = perc_list, y = p_value_list),
                row=i + 1, col = 1
                )
            fig.add_hline(y = 0.05, line_width=1, line_dash='solid', line_color='red', row=i + 1, col=1)
            if np.max(p_value_list) > 0.9:
                fig.add_hline(y = 0.95, line_width=1, line_dash='solid', line_color='red', row=i + 1, col=1)
            if np.max(p_value_list) > 0.45:
                fig.add_hline(y = 0.5, line_width=1, line_dash='dash', line_color='red', row=i + 1, col=1)
            fig.layout.annotations[i].update(text = f'Изменение p-value при делении на контрольную и тестовую группу<br>по оценке {rating_threshold} для параметра {feature}',
                font_color = 'blue', font_size = 22)
            fig.update_yaxes(range=[-0.05, np.max(p_value_list) + 0.1], row=i + 1, col=1)
            fig.update_xaxes(title='Порог деления параметра на хороший и плохой в процентилях', title_font=dict(size=18, color='green'), col=1, row=i + 1)
            fig.update_yaxes(title='p-value', title_font=dict(size=18, color='green'), col=1, row=i + 1)
        fig.update_layout(height=2500, width=1000, showlegend=False)
        fig.show()
    else:
        fig = make_subplots(rows = 6, cols = 1, subplot_titles = ('1', '2', '3', '4', '5', '6'))
        answers_dict = {1: 'Недозвоны, обрывы при звонках',
                2: 'Время ожидания гудков при звонке',
                3: 'Плохое качество связи в зданиях',
                4: 'Медленный мобильный Интернет',
                5: 'Медленная загрузка видео',
                6: 'Затрудняюсь ответить',
                7: 'Свой вариант'}
        for i, answer in enumerate(range(1, 7)):
            feature = df.columns[column_idx]
            if answer == 6:
                subplot_text = f'Изменение p-value при делении на контрольную и тестовую группу<br>по ответам "Затрудняюсь ответить" и "Свой вариант"<br>для параметра {feature}'
                control_df = df.loc[(df.iloc[:, target_feature_idx] == 6) | (df.iloc[:, target_feature_idx] == 7), :]
                test_df = df.loc[~df['user_id'].isin(control_df['user_id'])]
            else:
                subplot_text = f'Изменение p-value при делении на контрольную и тестовую группу<br>по ответу {answers_dict[i + 1]}<br>для параметра {feature}'
                control_df = df.loc[(df.iloc[:, target_feature_idx] == answer), :]
                test_df = df.loc[~df['user_id'].isin(control_df['user_id'])]
            x = control_df[feature]
            y = test_df[feature]
            p_value_list, perc_list = [], []
            for perc in np.linspace(0.95, 0.05, 19):
                perc = round(perc, 2)
                if hypotheses_type == 1:
                    good_threshold = df[feature].quantile(perc)
                    # good_threshold = min(x.quantile(perc), y.quantile(perc))
                    control_df['good/bad param'] = control_df[feature].apply(lambda x: 1 if x >= good_threshold else 0)
                    test_df['good/bad param'] = test_df[feature].apply(lambda x: 1 if x >= good_threshold else 0)
                    perc_list.append(perc)
                else:
                    good_threshold = df[feature].quantile(1 - perc)
                    # good_threshold = max(x.quantile(1 - perc), y.quantile(1 - perc))
                    control_df['good/bad param'] = control_df[feature].apply(lambda x: 1 if x <= good_threshold else 0)
                    test_df['good/bad param'] = test_df[feature].apply(lambda x: 1 if x <= good_threshold else 0)
                    perc_list.append(1 - perc)
                p_old = func(control_df['good/bad param'])
                p_new = func(test_df['good/bad param'])
                diff_base = p_new - p_old
                differences = np.zeros((1, N_TRIAL))
                for j in range(0, N_TRIAL):
                    if n == -1:
                        n_control = len(x)
                        n_test = len(y)
                    else:
                        n_control = n
                        n_test = n
                    s1 = random.choices(control_df['good/bad param'].to_numpy(), k = n_control)
                    s2 = random.choices(test_df['good/bad param'].to_numpy(), k = n_test)
                    p1 = func(s1)
                    p2 = func(s2)
                    differences[0][j] = p2 - p1
                differences_cent = differences - np.mean(differences)
                if alternative_hyp == 'more':
                    p_value = np.sum(differences_cent > diff_base) / N_TRIAL
                else:
                    p_value = np.sum(differences_cent < diff_base) / N_TRIAL
                p_value_list.append(p_value)
            fig.add_trace(
                go.Scatter(x = perc_list, y = p_value_list),
                row=i + 1, col = 1
                )
            fig.add_hline(y = 0.05, line_width=1, line_dash='solid', line_color='red', row=i + 1, col=1)
            if np.max(p_value_list) > 0.9:
                fig.add_hline(y = 0.95, line_width=1, line_dash='solid', line_color='red', row=i + 1, col=1)
            if np.max(p_value_list) > 0.45:
                fig.add_hline(y = 0.5, line_width=1, line_dash='dash', line_color='red', row=i + 1, col=1)
            fig.layout.annotations[i].update(text = subplot_text, font_color = 'blue', font_size = 22)
            fig.update_yaxes(range=[-0.05, np.max(p_value_list) + 0.1], row=i + 1, col=1)
            fig.update_xaxes(title='Порог деления параметра на хороший и плохой в процентилях', title_font=dict(size=18, color='green'), col=1, row=i + 1)
            fig.update_yaxes(title='p-value', title_font=dict(size=18, color='green'), col=1, row=i + 1)
        fig.update_layout(height=3000, width=1000, showlegend=False)
        fig.show()



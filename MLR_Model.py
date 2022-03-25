import pandas as pd
import numpy as np
from myClasses import Fixture_DF, Standings_V2, list_of_specific_files


# def data_for_train_test(game):
#     ys = year_season()
#     year = ["_" + x[-7:-5] + "_" + x[-2:] for x in ys]
#
#     df = pd.DataFrame()
#     for ind, i in enumerate(ys):
#
#         data_orig = pd.read_excel(rf"C:\Users\sabzu\Documents\All EPL Project Files\Fixtures\{i}.xlsx", header=1)
#         # data_h = data_orig[data_orig["Wk"] <= game]
#
#         fix = Fixture_DF(data_orig)
#         s = Standings_V2(fix, game)
#         # s = Standings(fix)
#         halfway = s.standings
#         halfway["Team"] = halfway["Team"].astype(str) + str(year[ind])
#
#         rank = list(range(1, 21))
#         halfway["rank"] = rank
#
#         fix1 = Fixture_DF(data_orig)
#         st = Standings_V2(fix1, 38)
#         # st = Standings(fix1)
#
#         full = st.standings
#         full["Team"] = full["Team"].astype(str) + str(year[ind])
#         rank = list(range(1, 21))
#         full["rank"] = rank
#
#         endRank = []
#         for i in halfway["Team"]:
#             q = full[full["Team"] == i]
#             w = int(q["rank"])
#             endRank.append(w)
#         halfway["EndRank"] = endRank
#
#         df = halfway if len(df) == 0 else pd.concat([df, halfway])
#
#     df = df.convert_dtypes().reset_index(drop=True)
#     # df.to_csv("C:\\Users\\sabzu\\Documents\\PremierStreamlit\\Test_Train_datasets\\AggStandingsAtGameX.csv", header=True)
#     if game < 10:
#         game_string = '0' + str(game)
#         df.to_csv(fr"C:\Users\sabzu\Documents\PremierLeagueStreamlitProject\PremierLeagueStreamlit\Storage_of_Historical_Standings\game{game_string}_standings.csv", header=True)
#     else:
#         df.to_csv(fr"C:\Users\sabzu\Documents\PremierLeagueStreamlitProject\PremierLeagueStreamlit\Storage_of_Historical_Standings\game{game}_standings.csv", header=True)
#     return df


# This will have to be updated when there is a new current season
def current_season_data_for_model(game, df):
    current = df

    fix2 = Fixture_DF(current)
    stand = Standings_V2(fix2, game)

    week_break = stand.standings
    week_break["Team"] = week_break["Team"].astype(str) + str("_21_22")  # This will need a year change when new season comes
    rank = list(range(1, 21))
    week_break["rank"] = rank
    week_break = week_break.convert_dtypes()

    if np.mean(week_break["MP"]) == float(game).__round__(2):
        if game < 10:
            game_string = '0' + str(game)
            week_break.to_csv(fr"C:\Users\sabzu\Documents\PremierLeagueStreamlitProject\PremierLeagueStreamlit\Storage_of_Current_Seasons_Standings\game{game_string}_Standings.csv", index=False)
        else:
            week_break.to_csv(fr"C:\Users\sabzu\Documents\PremierLeagueStreamlitProject\PremierLeagueStreamlit\Storage_of_Current_Seasons_Standings\game{game}_Standings.csv", index=False)

    return week_break


def predicted_PointsAndStandingsByModel(game):
    for file_number in range(len(list_of_specific_files(r"C:\Users\sabzu\Documents\PremierLeagueStreamlitProject\PremierLeagueStreamlit\Storage_of_Historical_Standings"))):
        game_in_file = list_of_specific_files(r"C:\Users\sabzu\Documents\PremierLeagueStreamlitProject\PremierLeagueStreamlit\Storage_of_Historical_Standings")[file_number]
        if game < 10:
            gs = '0' + str(game)
            if (str(gs)) == game_in_file[4:6]:
                df = pd.read_csv(
                    rf"C:\Users\sabzu\Documents\PremierLeagueStreamlitProject\PremierLeagueStreamlit\Storage_of_Historical_Standings\{game_in_file}",
                    index_col=0)
                skip_historical = True
                break
        elif (str(game)) == game_in_file[4:6]:
            df = pd.read_csv(
                rf"C:\Users\sabzu\Documents\PremierLeagueStreamlitProject\PremierLeagueStreamlit\Storage_of_Historical_Standings\{game_in_file}",
                index_col=0)
            skip_historical = True
            break
        skip_historical = False
    if not skip_historical:
        df = current_season_data_for_model(game, df)

    # Model
    X = df.iloc[:, [4,6,7]].values
    y = df.iloc[:, 5].values

    from sklearn.model_selection import train_test_split
    # print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)
    # print(len(X_train))

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    MLR = LinearRegression()
    MLR.fit(X_train, y_train)
    c_y_prediction = MLR.predict(X_test)

    # print("Coeff:", MLR.coef_)
    # print("Intercept", MLR.intercept_)
    # print("MSE", mean_squared_error(y_test, c_y_prediction))
    print("R2", r2_score(y_test, c_y_prediction))

    for file_number in range(
            len(list_of_specific_files(
                r"C:\Users\sabzu\Documents\PremierLeagueStreamlitProject\PremierLeagueStreamlit\Storage_of_Current_Seasons_Standings"))):
        game_in_file = \
            list_of_specific_files(r"C:\Users\sabzu\Documents\PremierLeagueStreamlitProject\PremierLeagueStreamlit\Storage_of_Current_Seasons_Standings")[
                file_number]
        if game < 10:
            gs = '0' + str(game)
            if (str(gs)) == game_in_file[4:6]:
                current_year_prediction = pd.read_csv(
                    rf"C:\Users\sabzu\Documents\PremierLeagueStreamlitProject\PremierLeagueStreamlit\Storage_of_Current_Seasons_Standings\{game_in_file}")
                skip_current = True
                break
        elif (str(game)) == game_in_file[4:6]:
            current_year_prediction = pd.read_csv(
                rf"C:\Users\sabzu\Documents\PremierLeagueStreamlitProject\PremierLeagueStreamlit\Storage_of_Current_Seasons_Standings\{game_in_file}")
            skip_current = True
            break
        skip_current = False
    if not skip_current:
        current_year_prediction = current_season_data_for_model(game, current_year_prediction)

    # current_year_prediction = current_season_data_for_model(game)
    explanatory = current_year_prediction.iloc[:, [4,6,7]].values
    dependent = current_year_prediction.iloc[:, 5].values
    c_y_prediction = MLR.predict(explanatory)

    predicted_standings = pd.DataFrame()
    predicted_standings["Team"] = current_year_prediction["Team"]
    predicted_standings["xPoints"] = c_y_prediction.round(0)
    predicted_standings["Actual Points"] = dependent
    predicted_standings["Performance"] = predicted_standings["Actual Points"] - predicted_standings["xPoints"]
    predicted_standings = predicted_standings.sort_values("xPoints", ascending=False).reset_index(drop=True)
    # print((sum(predicted_standings["Performance"]**2)/20)**.5)
    return predicted_standings


# Same as data_for_train_test except there is no CSV
# def allTimeStandingsDataAtGameX(game):
#     ys = year_season()
#     year = ["_" + x[-7:-5] + "_" + x[-2:] for x in ys]
#
#     df = pd.DataFrame()
#     for ind, i in enumerate(ys):
#
#         data_orig = pd.read_excel(rf"C:\Users\sabzu\Documents\All EPL Project Files\Fixtures\{i}.xlsx", header=1)
#         # data_h = data_orig[data_orig["Wk"] <= game]
#
#         fix = Fixture_DF(data_orig)
#         s = Standings_V2(fix, game)
#         # s = Standings(fix)
#         halfway = s.standings
#         halfway["Team"] = halfway["Team"].astype(str) + str(year[ind])
#
#         rank = list(range(1, 21))
#         halfway["rank"] = rank
#
#         fix1 = Fixture_DF(data_orig)
#         st = Standings_V2(fix1, 38)
#         # st = Standings(fix1)
#
#         full = st.standings
#         full["Team"] = full["Team"].astype(str) + str(year[ind])
#         rank = list(range(1, 21))
#         full["rank"] = rank
#
#         endRank = []
#         for i in halfway["Team"]:
#             q = full[full["Team"] == i]
#             w = int(q["rank"])
#             endRank.append(w)
#         halfway["EndRank"] = endRank
#
#         df = halfway if len(df) == 0 else pd.concat([df, halfway])
#
#     df = df.convert_dtypes().reset_index(drop=True)
#     return df

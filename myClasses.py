import pandas as pd
import numpy as np
import os


# @sl.cache
def list_of_specific_files(r_then_file_directory):
    files = os.listdir(f"{r_then_file_directory}")
    return list(files)

# @sl.cache(allow_output_mutation=True)
# def year_season():
#     year = []
#     for file_number in range(len(list_of_specific_files(r"C:\Users\sabzu\Documents\All EPL Project Files\Fixtures"))):
#         year_in_file = list_of_specific_files(r"C:\Users\sabzu\Documents\All EPL Project Files\Fixtures")[file_number]
#         split_year_in_file = year_in_file.split(" ")
#         for i in split_year_in_file:
#             if 'xlsx' in i:
#                 y = i.split(".")
#                 year.append(y[0])
#     year.pop(-1)
#     return year


# def data_for_train_test(game):
#     ys = year_season()
#     year = ["_" + x[-7:-5] + "_" + x[-2:] for x in ys]
#
#     df = pd.DataFrame()
#     for ind, i in enumerate(ys):
#
#         data_orig = pd.read_excel(rf"C:\Users\sabzu\Documents\All EPL Project Files\Fixtures\{i}.xlsx", header=1)
#         data_h = data_orig[data_orig["Wk"] <= game]
#
#         fix = Fixture_DF(data_h)
#         s = Standings(fix)
#
#         halfway = s.standings
#         halfway["Team"] = halfway["Team"].astype(str) + str(year[ind])
#
#         rank = [x for x in range(1, 21)]
#         halfway["rank"] = rank
#
#         fix1 = Fixture_DF(data_orig)
#         st = Standings(fix1)
#
#         full = st.standings
#         full["Team"] = full["Team"].astype(str) + str(year[ind])
#         rank = [x for x in range(1, 21)]
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
#     df = df.convert_dtypes()
#     df.to_csv("C:/Users/sabzu/Documents/PremierStreamlit/Test_Train_datasets/Train.csv", header=True)
#     return df


# def current_season_data_for_model(game):
#     current = pd.read_excel('C:\\Users\\sabzu\\Documents\\All EPL Project Files\\Fixtures\\Fixtures_2021_2022.xlsx',header=1)
#     current = current[current["Wk"] <= game]
#
#     fix2 = Fixture_DF(current)
#     stand = Standings(fix2)
#
#     week_break = stand.standings
#     week_break["Team"] = week_break["Team"].astype(str) + str("_21_22")
#     rank = [x for x in range(1, 21)]
#     week_break["rank"] = rank
#     c_teams = list(week_break["Team"])
#     week_break = week_break.convert_dtypes()
#     week_break.to_csv("C:/Users/sabzu/Documents/PremierStreamlit/Test_Train_datasets/Current.csv", index=False)
#     return week_break



class Fixture_DF:
    def __init__(self, data):

        self._fixture_list = data
        fixture_list = pd.DataFrame(self._fixture_list)

        fixture_list = fixture_list[["Day", "Date", "Home", "Score", "Away", "Referee"]]
        fixture_list = fixture_list.dropna(subset=["Score"])
        fixture_list = fixture_list.reset_index(drop=True)
        self.fixture_list_df = fixture_list

        self.fixture_list_df["Winner"] = None
        self.fixture_list_df["Loser"] = None
        self._win_loss()
        unique_home_teams = list(self.fixture_list_df.Home.unique())
        unique_away_teams = list(self.fixture_list_df.Away.unique())
        self.unique_teams_grouped = unique_home_teams + unique_away_teams
        self.unique_teams_grouped = np.array(self.unique_teams_grouped)
        self.unique_teams_grouped = np.unique(self.unique_teams_grouped)
        # self.team_list = list(self.fixture_list_df.Home.unique()) + list(self.fixture_list_df.Away.unique())
        self.team_list = self.unique_teams_grouped.copy()
        self.team_list.sort()

    def _win_loss(self):
        for i in range(len(self.fixture_list_df)):
            if self.fixture_list_df.loc[i, "Score"][0] > self.fixture_list_df.loc[i, "Score"][2]:
                self.fixture_list_df.loc[i, "Winner"] = self.fixture_list_df.loc[i, "Home"]
                self.fixture_list_df.loc[i, "Loser"] = self.fixture_list_df.loc[i, "Away"]
            elif self.fixture_list_df.loc[i, "Score"][0] < self.fixture_list_df.loc[i, "Score"][2]:
                self.fixture_list_df.loc[i, "Winner"] = self.fixture_list_df.loc[i, "Away"]
                self.fixture_list_df.loc[i, "Loser"] = self.fixture_list_df.loc[i, "Home"]
            else:
                self.fixture_list_df.loc[i, "Winner"] = "Tie"
                self.fixture_list_df.loc[i, "Loser"] = "Tie"


class Standings_V2:
    def __init__(self, fixture, game=None):
        self._fixt = fixture
        self._game = game
        self.standings = pd.DataFrame()
        self.standings["Team"] = self._fixt.team_list
        self.standings["MP"] = None
        self.standings["W"] = None
        self.standings["D"] = None
        self.standings["L"] = None
        self.standings["Pts"] = None
        self.standings["GF"] = None
        self.standings["GA"] = None
        self.standings["GD"] = None

        self._win()
        self._loss()
        self._draw()
        self._matches_played()
        self._points()
        self._goals_for()
        self._goals_against()
        self._goal_difference()
        self.standings = (self.standings.sort_values(["Pts", "GD"], ascending=False)).reset_index(drop=True)

    def _matches_played(self):
        self.standings["MP"] = self.standings["W"] + self.standings["D"] + self.standings["L"]

    def _win(self):
        for n, i in enumerate(self._fixt.team_list):
            t = (self._fixt.fixture_list_df[
                (self._fixt.fixture_list_df["Home"] == f"{i}") | (self._fixt.fixture_list_df["Away"] == f"{i}")])
            t = t.reset_index(drop=True)
            if self._game is not None:
                t = t.iloc[:self._game, :]
            wins = len(t[t["Winner"] == f"{i}"])
            self.standings.loc[n, "W"] = wins

    def _loss(self):
        for n, i in enumerate(self._fixt.team_list):
            t = (self._fixt.fixture_list_df[(self._fixt.fixture_list_df["Home"] == f"{i}") | (self._fixt.fixture_list_df["Away"] == f"{i}")])
            t = t.reset_index(drop=True)
            if self._game is not None:
                t = t.iloc[:self._game, :]
            wins = len(t[t["Loser"] == f"{i}"])
            self.standings.loc[n, "L"] = wins

    def _draw(self):
        for n, i in enumerate(self._fixt.team_list):
            t = (self._fixt.fixture_list_df[(self._fixt.fixture_list_df["Home"] == f"{i}") | (self._fixt.fixture_list_df["Away"] == f"{i}")])
            t = t.reset_index(drop=True)
            if self._game is not None:
                t = t.iloc[:self._game, :]
            wins = len(t[t["Winner"] == "Tie"])
            self.standings.loc[n, "D"] = wins

    def _points(self):
        self.standings["Pts"] = (self.standings["W"] * 3) + (self.standings["D"])

    def _goals_for(self):
        for n, i in enumerate(self._fixt.team_list):
            goals = 0

            # Home Goals For
            h_a = (self._fixt.fixture_list_df[
                (self._fixt.fixture_list_df["Home"] == f"{i}") | (self._fixt.fixture_list_df["Away"] == f"{i}")])
            h = h_a.reset_index(drop=True)
            if self._game is not None:
                h = h.iloc[:self._game, :]
            h = (h[(h["Home"] == f"{i}")])
            h = h.reset_index(drop=True)
            for row in range(len(h)):
                s = h["Score"][row]
                goals += int(s[0])

            # Away Goals For
            a = h_a.reset_index(drop=True)
            if self._game is not None:
                a = a.iloc[:self._game, :]
            a = (a[(a["Away"] == f"{i}")])
            a = a.reset_index(drop=True)
            for r in range(len(a)):
                s = a["Score"][r]
                goals += int(s[2])
            self.standings.loc[n, "GF"] = goals

    def _goals_against(self):
        for n, i in enumerate(self._fixt.team_list):
            goals = 0

            # Home Goals Against
            h_a = (self._fixt.fixture_list_df[
                (self._fixt.fixture_list_df["Home"] == f"{i}") | (self._fixt.fixture_list_df["Away"] == f"{i}")])
            h = h_a.reset_index(drop=True)
            if self._game is not None:
                h = h.iloc[:self._game, :]
            h = (h[(h["Home"] == f"{i}")])
            h = h.reset_index(drop=True)
            for row in range(len(h)):
                s = h["Score"][row]
                goals += int(s[2])

            # Away Goals Against
            a = h_a.reset_index(drop=True)
            if self._game is not None:
                a = a.iloc[:self._game, :]
            a = (a[(a["Away"] == f"{i}")])
            a = a.reset_index(drop=True)
            for r in range(len(a)):
                s = a["Score"][r]
                goals += int(s[0])
            self.standings.loc[n, "GA"] = goals

    def _goal_difference(self):
        self.standings["GD"] = self.standings["GF"] - self.standings["GA"]

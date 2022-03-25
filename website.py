import streamlit as st
from myClasses import Fixture_DF, Standings_V2, list_of_specific_files
from MLR_Model import current_season_data_for_model
import streamlit as sl
import pandas as pd
import numpy as np
from scipy import stats
from datetime import date, datetime, timedelta
from matplotlib.figure import Figure
import seaborn as sns


@sl.cache()  # Add home rank column and an away rank column
def teamRanks(dataframe):
    home = []
    away = []
    for ht in dataframe["Home"]:
        hrank = league_standings.index[league_standings["Team"] == ht].to_list()
        home.append(hrank[0])
    for at in dataframe["Away"]:
        arank = league_standings.index[league_standings["Team"] == at].to_list()
        away.append(arank[0])

    dataframe["H-Rank"] = home
    dataframe["A-Rank"] = away

    dataframe = dataframe.loc[:, ["Day", "Date", "Home", "Away", "H-Rank", "A-Rank"]]
    return dataframe


season = "_21_22"
g_files = (list_of_specific_files(r"C:\Users\sabzu\Documents\PremierLeagueStreamlitProject\PremierLeagueStreamlit\Storage_of_Current_Seasons_Standings"))
games_played = len(g_files)

score_fixt_df = pd.read_html("https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures")
score_fixt_df = score_fixt_df[0]
fixtures = Fixture_DF(score_fixt_df)
std = Standings_V2(fixtures)

team_list = fixtures.unique_teams_grouped.tolist()

# Check to see if the standings are up to date for the minimum games played by all teams
games = min(std.standings["MP"])
mostGames = max(std.standings["MP"])
if games_played != games:  # If not updated, run function to update
    current_season_data_for_model(games, score_fixt_df)

sl.set_page_config(layout="wide")
sl.title("English Premier League Statistics and Analytics")
with sl.sidebar:
    rad = sl.radio("Selection", ("Current", "League", "Team History"))

if rad == "Current":
    with sl.sidebar:
        g = sl.number_input("Games Played", min_value=1, max_value=mostGames, value=mostGames)
        st = Standings_V2(fixtures, int(g))
    league_standings = st.standings
    league_standings["Rank"] = [r for r in range(1, 21)]
    league_standings.set_index('Rank', drop=True, inplace=True)
    sl.table(league_standings)  # Current Standings
    sl.write("Games in the next 7 days:")

    if np.mean(league_standings["MP"]) != 38:
        # Upcoming fixtures within 7 days of the current date
        today = date.today()
        today = datetime.strptime(str(today), '%Y-%m-%d')
        end_date = today + timedelta(days=7)
        score_fixt_df["Date"] = pd.to_datetime(score_fixt_df["Date"])
        score_fixt_df["Date"] = pd.to_datetime(score_fixt_df["Date"], '%Y-%m-%d')
        df = (score_fixt_df[(score_fixt_df["Date"] >= today) & (score_fixt_df["Date"] <= end_date)])
        df["Date"] = pd.to_datetime(df["Date"], format='%Y-%m-%d %H:%M:%S').dt.strftime('%Y-%m-%d')
        sl.table(teamRanks(df))

elif rad == "League":
    with sl.sidebar:
        if games_played > 19:
            statSelection = sl.selectbox("Stat Choice", ("Percentile", "Similarity", "Remaining Schedule"))
        else:
            statSelection = sl.selectbox("Stat Choice", ("Percentile", "Similarity"))

    if statSelection == "Percentile":
        with sl.sidebar:
            mp = int(sl.number_input(label="Game Number", min_value=1, max_value=mostGames, value=games_played))
            # mp = int(sl.number_input(min_value=1, max_value=38, label="Games", value=games_played))
            stat = sl.selectbox(label="Choose your stat", options=["Pts", "W", "D", "L", "GF", "GA", "GD"])
        selected_games = mp

        if mp < 10:
            mp = "0" + str(mp)

        if selected_games <= games_played:
            sl.write(
                f"Since the 2016/2017 season, what is the percentile for a current teams {stat} up to {mp} games played?")

            c = pd.read_csv(
                fr"C:\Users\sabzu\Documents\PremierLeagueStreamlitProject\PremierLeagueStreamlit\Storage_of_Current_Seasons_Standings\game{mp}_Standings.csv")
            df2 = pd.read_csv(
                fr"C:\Users\sabzu\Documents\PremierLeagueStreamlitProject\PremierLeagueStreamlit\Storage_of_Historical_Standings\game{mp}_standings.csv",
                index_col=0)
            std = np.std(df2[stat])
            ave = np.mean(df2[stat])

            percentitleDF = pd.DataFrame()
            Team = []
            statistic = []
            percent = []

            for id, t in enumerate(c["Team"]):
                Team.append(t)
                statistic.append(c[stat][id])
                zscore = (c[stat][id] - ave) / std
                percent.append((stats.norm.cdf(zscore)) * 100)

            percentitleDF["Team"] = Team
            percentitleDF[stat] = statistic
            percentitleDF["Percentile"] = percent
            percentitleDF.sort_values('Percentile', ascending=False, inplace=True)
            percentitleDF["Rank"] = [x for x in range(1, 21)]
            percentitleDF.set_index("Rank", inplace=True)
            sl.table(percentitleDF)

        else:
            teamsPlayedMoreThanMin_df = current_season_data_for_model(selected_games, score_fixt_df)
            teamsPlayedMoreThanMin_df = teamsPlayedMoreThanMin_df[teamsPlayedMoreThanMin_df["MP"] == selected_games]
            teamsPlayedMoreThanMin_df.reset_index(drop=True, inplace=True)
            df2 = pd.read_csv(
                fr"C:\Users\sabzu\Documents\PremierLeagueStreamlitProject\PremierLeagueStreamlit\Storage_of_Historical_Standings\game{mp}_standings.csv",
                index_col=0)
            std = np.std(df2[stat])
            ave = np.mean(df2[stat])

            percentitleDF = pd.DataFrame()
            Team = []
            statistic = []
            percent = []

            for id, t in enumerate(teamsPlayedMoreThanMin_df["Team"]):
                Team.append(t)
                statistic.append(teamsPlayedMoreThanMin_df[stat][id])
                zscore = (teamsPlayedMoreThanMin_df[stat][id] - ave) / std
                percent.append((stats.norm.cdf(zscore)) * 100)

            percentitleDF["Team"] = Team
            percentitleDF[stat] = statistic
            percentitleDF["Percentile"] = percent
            percentitleDF.sort_values('Percentile', ascending=False, inplace=True)
            percentitleDF["Rank"] = [x for x in range(1, len(teamsPlayedMoreThanMin_df) + 1)]
            percentitleDF.set_index("Rank", inplace=True)
            sl.table(percentitleDF)

        top_or_bottom = sl.selectbox(label="Top or bottom 10%", options=["Top 10%", "Bottom 10%"])
        sl.write(f"The {top_or_bottom} for {stat} at matchweek {int(mp)} since 2016/2017")

        if top_or_bottom == "Top 10%":
            t10p = pd.read_csv(
                rf"C:\Users\sabzu\Documents\PremierLeagueStreamlitProject\PremierLeagueStreamlit\AllTime\TopTenPercent\{stat}\TopTenPercentGame{mp}.csv",
                index_col=0)
            t10p["Rank"] = [x for x in range(1, len(t10p) + 1)]
            t10p.set_index("Rank", inplace=True)
            sl.table(t10p)
        else:
            b10p = pd.read_csv(
                rf"C:\Users\sabzu\Documents\PremierLeagueStreamlitProject\PremierLeagueStreamlit\AllTime\BottomTenPercent\{stat}\BottomTenPercentGame{mp}.csv",
                index_col=0)
            b10p["Rank"] = [x for x in range(1, len(b10p) + 1)]
            b10p.set_index("Rank", inplace=True)
            sl.table(b10p)

    if statSelection == "Similarity":
        with sl.sidebar:
            mp = int(sl.number_input(label="Game Number", min_value=1, max_value=mostGames, value=games_played))
            selected_games = mp

        if mp < 10:
            mp = '0' + str(mp)

        len_similarity_storage = len(list_of_specific_files(
            r"C:\Users\sabzu\Documents\PremierLeagueStreamlitProject\PremierLeagueStreamlit\Similarity_CSV_files\Similarity_by_Week"))

        if selected_games in range(1, len_similarity_storage + 1):
            euclid_df = pd.read_csv(
                fr"C:\Users\sabzu\Documents\PremierLeagueStreamlitProject\PremierLeagueStreamlit\Similarity_CSV_files\Similarity_by_Week\game{mp}.csv",
                index_col=0)
            currentTeamsStandings = pd.read_csv(
                fr"C:\Users\sabzu\Documents\PremierLeagueStreamlitProject\PremierLeagueStreamlit\Storage_of_Current_Seasons_Standings\game{mp}_Standings.csv")
            historicalStandings = pd.read_csv(
                fr"C:\Users\sabzu\Documents\PremierLeagueStreamlitProject\PremierLeagueStreamlit\Storage_of_Historical_Standings\game{mp}_standings.csv",
                index_col=0)
            full_season = (pd.read_csv(
                r"C:\Users\sabzu\Documents\PremierLeagueStreamlitProject\PremierLeagueStreamlit\Storage_of_Historical_Standings\game38_standings.csv",
                index_col=0))

            similarity_df = pd.DataFrame(columns=["Team", "MostSimilar", "SimilarityMeasure"])
            m_sim = []
            SimilarityMeasure = []
            c_teams = euclid_df.columns

            for c in euclid_df.columns:
                minn = min(euclid_df[c])
                sim = list(euclid_df[euclid_df[c] == minn].index)
                m_sim.append(sim[0])
                SimilarityMeasure.append(minn)
            similarity_df["Team"] = c_teams
            similarity_df["MostSimilar"] = m_sim
            similarity_df["SimilarityMeasure"] = SimilarityMeasure
            sl.table(similarity_df)

            selection = sl.selectbox("Choose a team", similarity_df["Team"])
            selected_team = currentTeamsStandings[currentTeamsStandings["Team"] == selection].copy().reset_index(
                drop=True)
            selected_team["EndRank"] = np.nan

            a = euclid_df[selection].sort_values(ascending=True)
            orig_team = currentTeamsStandings[currentTeamsStandings["Team"] == selection].copy()
            orig_team["EndRank"] = np.nan
            orig_team["Similarity"] = np.nan

            first_team = historicalStandings[historicalStandings["Team"] == a.index[0]].copy()
            first_team["Similarity"] = a[0]
            comp_df = pd.concat([orig_team, first_team])

            for num in range(1, 10):
                if a[num] <= (a[0] * 1.5):
                    other_team = historicalStandings[historicalStandings["Team"] == a.index[num]].copy()
                    other_team["Similarity"] = a[num]
                    comp_df = pd.concat([comp_df, other_team]).reset_index(drop=True)
                    if num == 9:
                        sl.write(comp_df.reset_index(drop=True))
                        # print(comp_df.reset_index(drop=True))
                else:
                    sl.write(comp_df.reset_index(drop=True))
                    # print(comp_df.reset_index(drop=True))
                    break
            ct = list((comp_df["Team"].reset_index(drop=True)))
            ct.pop(0)

            ndf = pd.DataFrame()
            for i in ct:
                tm = full_season[full_season["Team"] == i]
                ndf = ndf.append(tm)
            # print(ndf.reset_index(drop=True))
            sl.table(ndf)

        else:
            if selected_games == games_played:
                df = pd.read_csv(
                    rf"C:\Users\sabzu\Documents\PremierLeagueStreamlitProject\PremierLeagueStreamlit\Storage_of_Historical_Standings\game{mp}_standings.csv",
                    index_col=0)
                week_break = pd.read_csv(
                    rf"C:\Users\sabzu\Documents\PremierLeagueStreamlitProject\PremierLeagueStreamlit\Storage_of_Current_Seasons_Standings\game{mp}_Standings.csv")

                c_teams = list(week_break["Team"])
                list_dict = {t: [] for t in range(len(week_break))}
                game_played_reference = 0
                df2_storage = {}
                for row in range(len(week_break)):
                    if week_break["MP"][row] != selected_games:
                        game_played = week_break["MP"][row]
                        if game_played != game_played_reference:
                            if game_played in df2_storage:
                                df_2 = df2_storage[game_played]
                            else:
                                df_2 = pd.read_csv(
                                    fr"C:\Users\sabzu\Documents\PremierLeagueStreamlitProject\PremierLeagueStreamlit\Storage_of_Historical_Standings\game{mp}_standings.csv",
                                    index_col=0)
                                df2_storage[game_played] = df_2

                        for comp in range(len(df_2)):
                            t_euclid = ((((week_break.iloc[row, 5] - df_2.iloc[comp, 5]) ** 2) + (
                                    (week_break.iloc[row, 6] - df_2.iloc[comp, 6]) ** 2)
                                         + ((week_break.iloc[row, 7] - df_2.iloc[comp, 7]) ** 2)) ** .5)
                            list_dict[row].append(t_euclid)
                        game_played_reference = game_played

                    else:
                        for comp in range(len(df)):
                            t_euclid = ((((week_break.iloc[row, 5] - df.iloc[comp, 5]) ** 2) + (
                                    (week_break.iloc[row, 6] - df.iloc[comp, 6]) ** 2) + (
                                                 (week_break.iloc[row, 7] - df.iloc[comp, 7]) ** 2)) ** .5)
                            list_dict[row].append(t_euclid)

                df_labels = list(df.iloc[:len(df), 0])
                euclid_df = pd.DataFrame()
                euclid_df["Team"] = df_labels
                key = list(list_dict.keys())
                for label, team in enumerate(list_dict):
                    euclid_df[key[label]] = list_dict[team]
                euclid_df = euclid_df.set_index("Team")

                for cid, n in enumerate(c_teams):
                    euclid_df.rename(columns={cid: n}, inplace=True)

                euclid_df.to_csv(
                    "C:\\Users\\sabzu\\Documents\\PremierLeagueStreamlitProject\\PremierLeagueStreamlit\\Similarity_CSV_files\\current Season similarity.csv",
                    index=True)

                if np.mean(week_break["MP"]) == selected_games:
                    euclid_df.to_csv(
                        f"C:\\Users\\sabzu\Documents\\PremierLeagueStreamlitProject\\PremierLeagueStreamlit\\Similarity_CSV_files\\Similarity_by_Week\\game{mp}.csv",
                        index=True)

                similarity_df = pd.DataFrame(columns=["Team", "MostSimilar", "SimilarityMeasure"])
                m_sim = []
                SimilarityMeasure = []
                c_teams = euclid_df.columns

                for c in euclid_df.columns:
                    minn = min(euclid_df[c])
                    sim = list(euclid_df[euclid_df[c] == minn].index)
                    m_sim.append(sim[0])
                    SimilarityMeasure.append(minn)
                similarity_df["Team"] = c_teams
                similarity_df["MostSimilar"] = m_sim
                similarity_df["SimilarityMeasure"] = SimilarityMeasure
                sl.table(similarity_df)

                currentTeamsStandings = pd.read_csv(
                    fr"C:\Users\sabzu\Documents\PremierLeagueStreamlitProject\PremierLeagueStreamlit\Storage_of_Current_Seasons_Standings\game{mp}_Standings.csv")
                historicalStandings = pd.read_csv(
                    fr"C:\Users\sabzu\Documents\PremierLeagueStreamlitProject\PremierLeagueStreamlit\Storage_of_Historical_Standings\game{mp}_standings.csv",
                    index_col=0)
                full_season = (pd.read_csv(
                    r"C:\Users\sabzu\Documents\PremierLeagueStreamlitProject\PremierLeagueStreamlit\Storage_of_Historical_Standings\game38_standings.csv",
                    index_col=0))

                selection = sl.selectbox("Choose a team", similarity_df["Team"])
                selected_team = currentTeamsStandings[currentTeamsStandings["Team"] == selection].copy().reset_index(
                    drop=True)
                selected_team["EndRank"] = np.nan

                a = euclid_df[selection].sort_values(ascending=True)
                orig_team = currentTeamsStandings[currentTeamsStandings["Team"] == selection].copy()
                orig_team["EndRank"] = np.nan
                orig_team["Similarity"] = np.nan

                first_team = historicalStandings[historicalStandings["Team"] == a.index[0]].copy()
                first_team["Similarity"] = a[0]
                comp_df = pd.concat([orig_team, first_team])

                for num in range(1, 10):
                    if a[num] <= (a[0] * 1.5):
                        other_team = historicalStandings[historicalStandings["Team"] == a.index[num]].copy()
                        other_team["Similarity"] = a[num]
                        comp_df = pd.concat([comp_df, other_team]).reset_index(drop=True)
                        if num == 9:
                            sl.write(comp_df.reset_index(drop=True))
                            # print(comp_df.reset_index(drop=True))
                    else:
                        sl.write(comp_df.reset_index(drop=True))
                        # print(comp_df.reset_index(drop=True))
                        break
                ct = list((comp_df["Team"].reset_index(drop=True)))
                ct.pop(0)

                ndf = pd.DataFrame()
                for i in ct:
                    tm = full_season[full_season["Team"] == i]
                    ndf = ndf.append(tm)
                # print(ndf.reset_index(drop=True))
                sl.table(ndf)


            else:
                if selected_games > games_played:
                    df = pd.read_csv(
                        f"C:\\Users\\sabzu\\Documents\\PremierLeagueStreamlitProject\\PremierLeagueStreamlit\\Storage_of_Historical_Standings\\game{mp}_standings.csv",
                        index_col=0)

                    teamsPlayedMoreThanMin_df = current_season_data_for_model(selected_games, score_fixt_df)
                    teamsPlayedMoreThanMin_df = teamsPlayedMoreThanMin_df[
                        teamsPlayedMoreThanMin_df["MP"] == selected_games].copy()
                    teamsPlayedMoreThanMin_df.reset_index(drop=True, inplace=True)

                    c_teams = list(teamsPlayedMoreThanMin_df["Team"])
                    list_dict = {t: [] for t in range(len(teamsPlayedMoreThanMin_df))}
                    game_played_reference = 0
                    df2_storage = {}

                    for row in range(len(teamsPlayedMoreThanMin_df)):
                        for comp in range(len(df)):
                            t_euclid = ((((teamsPlayedMoreThanMin_df.iloc[row, 5] - df.iloc[comp, 5]) ** 2) + (
                                    (teamsPlayedMoreThanMin_df.iloc[row, 6] - df.iloc[comp, 6]) ** 2)
                                         + ((teamsPlayedMoreThanMin_df.iloc[row, 7] - df.iloc[comp, 7]) ** 2)) ** .5)
                            list_dict[row].append(t_euclid)

                    df_labels = list(df.iloc[:len(df), 0])
                    euclid_df = pd.DataFrame()
                    euclid_df["Team"] = df_labels
                    key = list(list_dict.keys())
                    for label, team in enumerate(list_dict):
                        euclid_df[key[label]] = list_dict[team]
                    euclid_df = euclid_df.set_index("Team")

                    for cid, n in enumerate(c_teams):
                        euclid_df.rename(columns={cid: n}, inplace=True)

                    similarity_df = pd.DataFrame(columns=["Team", "MostSimilar", "SimilarityMeasure"])
                    m_sim = []
                    SimilarityMeasure = []
                    c_teams = euclid_df.columns
                    for c in euclid_df.columns:
                        minn = min(euclid_df[c])
                        sim = list(euclid_df[euclid_df[c] == minn].index)
                        m_sim.append(sim[0])
                        SimilarityMeasure.append(minn)
                    similarity_df["Team"] = c_teams
                    similarity_df["MostSimilar"] = m_sim
                    similarity_df["SimilarityMeasure"] = SimilarityMeasure
                    sl.table(similarity_df)

                    historicalStandings = pd.read_csv(
                        fr"C:\Users\sabzu\Documents\PremierLeagueStreamlitProject\PremierLeagueStreamlit\Storage_of_Historical_Standings\game{mp}_standings.csv",
                        index_col=0)
                    full_season = (pd.read_csv(
                        r"C:\Users\sabzu\Documents\PremierLeagueStreamlitProject\PremierLeagueStreamlit\Storage_of_Historical_Standings\game38_standings.csv",
                        index_col=0))
                    selection = sl.selectbox("Choose a team", similarity_df["Team"])
                    selected_team = teamsPlayedMoreThanMin_df[
                        teamsPlayedMoreThanMin_df["Team"] == selection].copy().reset_index(
                        drop=True)
                    selected_team["EndRank"] = np.nan

                    a = euclid_df[selection].sort_values(ascending=True)
                    orig_team = teamsPlayedMoreThanMin_df[teamsPlayedMoreThanMin_df["Team"] == selection].copy()
                    orig_team["EndRank"] = np.nan
                    orig_team["Similarity"] = np.nan

                    first_team = historicalStandings[historicalStandings["Team"] == a.index[0]].copy()
                    first_team["Similarity"] = a[0]
                    comp_df = pd.concat([orig_team, first_team])

                    for num in range(1, 10):
                        if a[num] <= (a[0] * 1.5):
                            other_team = historicalStandings[historicalStandings["Team"] == a.index[num]].copy()
                            other_team["Similarity"] = a[num]
                            comp_df = pd.concat([comp_df, other_team]).reset_index(drop=True)
                            if num == 9:
                                sl.write(comp_df.reset_index(drop=True))
                        else:
                            sl.write(comp_df.reset_index(drop=True))
                            break
                    ct = list((comp_df["Team"].reset_index(drop=True)))
                    ct.pop(0)

                    ndf = pd.DataFrame()
                    for i in ct:
                        tm = full_season[full_season["Team"] == i]
                        ndf = ndf.append(tm)
                    sl.table(ndf)

    if statSelection == "Remaining Schedule":
        league_standings = std.standings
        league_standings["Rank"] = [r for r in range(1, 21)]
        league_standings.set_index('Rank', drop=True, inplace=True)

        today = date.today()
        today = datetime.strptime(str(today), '%Y-%m-%d')
        score_fixt_df["Date"] = pd.to_datetime(score_fixt_df["Date"])
        score_fixt_df["Date"] = pd.to_datetime(score_fixt_df["Date"], '%Y-%m-%d')
        df = (score_fixt_df[(score_fixt_df["Date"] >= today)])
        df.loc[:, "Date"] = pd.to_datetime(df["Date"], format='%Y-%m-%d %H:%M:%S').dt.strftime('%Y-%m-%d')
        df = df.loc[:, ["Day", "Date", "Home", "Away"]]

        df = (teamRanks(df))
        oppRankDf = pd.DataFrame()
        AveRank = []
        hrank = []
        arank = []
        for i in range(20):
            homeSched = df[(df["Home"] == team_list[i])]
            awaySched = df[(df["Away"] == team_list[i])]

            aveOppRank = ((np.mean(homeSched["A-Rank"]) + np.mean(awaySched["H-Rank"])) / 2).__round__(1)
            AveRank.append(aveOppRank)
            hr = np.mean(homeSched["A-Rank"]).__round__(1)
            hrank.append(hr)
            ar = np.mean(awaySched["H-Rank"]).__round__(1)
            arank.append(ar)

        oppRankDf["Team"] = team_list
        oppRankDf["OverallOppRank"] = AveRank
        oppRankDf["H-GameOppRk"] = hrank
        oppRankDf["A-GameOppRk"] = arank
        oppRankDf = oppRankDf.sort_values("OverallOppRank").reset_index(drop=True)
        sl.table(oppRankDf)

elif rad == "Team History":
    team = sl.selectbox(label="Select Team", options=fixtures.unique_teams_grouped)

    # Get the stat-line for only the selected team
    league_standings = std.standings
    t = league_standings[league_standings["Team"] == team].copy()
    t["Team"] = team + season
    played = t["MP"].reset_index(drop=True)
    played = played[0]

    played = sl.number_input(label="Select games played", min_value=1, max_value=played, value=played)
    # Get all the previous occurrences of selected team in the historical standings at game X
    team_points = []
    historic_std = pd.read_csv(
        fr"C:\Users\sabzu\Documents\PremierLeagueStreamlitProject\PremierLeagueStreamlit\Storage_of_Historical_Standings\game{int(played)}_standings.csv",
        index_col=0)
    mask = historic_std["Team"].str.contains(team, case=False, na=False)
    historic_std = historic_std[mask]
    all_selected_team_df = pd.concat([historic_std, t])

    stat = sl.selectbox(label="Choose your stat", options=["Pts", "W", "D", "L", "GF", "GA", "GD"])

    # Create the plot for whatever the selected stat is
    fig = Figure()
    ax = fig.subplots()
    chart = sns.barplot(x=all_selected_team_df["Team"], y=all_selected_team_df[stat], ax=ax).set(
        title=f"{stat} for {team} after {int(played)} games")
    value_labs = all_selected_team_df[stat].values

    ax.set_xlabel("Season")
    ax.set_xticklabels(labels=all_selected_team_df["Team"], rotation=30, size=10)
    ax.bar_label(container=ax.containers[0], labels=value_labs)
    sns.set(rc={'figure.figsize': (15, 8)})
    st.pyplot(fig)

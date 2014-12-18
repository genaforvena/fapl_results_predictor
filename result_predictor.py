import csv
import sys

import NaiveBayes

from sklearn import svm


__author__ = 'imozerov'

bayes = NaiveBayes.NaiveBayes()
svm = svm.SVC()


def get_team_names(csv_filename):
    teams = []
    with open(csv_filename, 'rt') as csvfile:
        reader = csv.reader(csvfile)
        iterator = iter(reader)
        next(iterator)
        for row in iterator:
            teams.append(row[2])
            teams.append(row[3])
    uniq_teams = set(teams)
    return uniq_teams


def get_data_list(csv_filename):
    data = []
    with open(csv_filename, 'rt') as csvfile:
        reader = csv.reader(csvfile)
        iterator = iter(reader)
        next(iterator)
        for row in iterator:
            data.append(row)
    return data


def get_game_result(row):
    game_result = "D"
    if row[6] == "A":
        game_result = "L"
    if row[6] == "H":
        game_result = "W"
    return game_result


def get_home_or_away(row, team):
    if row[3] == team:
        opponent = row[2]
        home_or_away = "A"
    else:
        opponent = row[3]
        home_or_away = "H"
    return home_or_away, opponent


def print_statistics_table(files, out_filename):
    for file in files:
        teams = get_team_names(file)
        data = get_data_list(file)
        with open(out_filename, "w+") as data_file:
            writer = csv.writer(data_file)
            writer.writerow(["team", "opponent", "date", "result", "H/A" "scored", "conceded",
                             "recent_form", "total_scored", "total_conceded", "avg_scrd", "avg_cons"])
            for team in teams:
                total_scored = 0
                total_conceded = 0
                games = 0
                recent_form = ""
                for row in data:
                    if row[2] == team or row[3] == team:
                        games += 1
                        game_result = get_game_result(row)
                        (home_or_away, opponent) = get_home_or_away(row, team)
                        date = row[1]
                        scored = row[4]
                        conceded = row[5]
                        writer.writerow([team, opponent, date, game_result, home_or_away,
                                         scored, conceded, recent_form,
                                         total_scored, total_conceded,
                                         total_scored / games, total_conceded / games])
                        total_scored += int(scored)
                        total_conceded += int(conceded)
                        if len(recent_form) < 5:
                            recent_form += game_result
                        else:
                            recent_form = recent_form[1:] + game_result


def train(filename):
    with open(filename, "rt") as data_file:
        reader = csv.reader(data_file)
        iterator = iter(reader)
        next(iterator)
        results = list(iterator)
        for result in results:
            teamA = result[0]
            teamB = result[1]
            date = result[2]
            teamA_data = result
            for row2 in results:
                if row2[0] == teamB and row2[2] == date:
                    teamB_data = row2
            data = teamA_data[4:] + teamB_data[4:]
            bayes.add_instances({'attributes': dict(enumerate(data)), 'label': result[3], 'cases': 1})
            svm.add({'attributes': dict(enumerate(data)), 'label': result[3], 'cases': 1})
        bayes.train()
        svm.train()


def predict(filename):
    with open(filename, "rt") as data_file:
        reader = csv.reader(data_file)
        iterator = iter(reader)
        next(iterator)
        results = list(iterator)
        for result in results:
            teamA = result[0]
            teamB = result[1]
            date = result[2]
            teamA_data = result
            for row2 in results:
                if row2[0] == teamB and row2[2] == date:
                    teamB_data = row2
            data = teamA_data[4:] + teamB_data[4:]
            try:
                print("bayes: " + bayes.predict({'attributes': dict(enumerate(data))}))
                print("svm: " + svm.predict({'attributes': dict(enumerate(data))}))
                print("actual: " + result[3])
            except:
                pass


if __name__ == "__main__":
    files = sys.argv[1:]
    train_data = "train.csv"
    test_data = "test.csv"
    print_statistics_table(files[:-1], train_data)
    train(train_data)
    print_statistics_table([files[-1]], test_data)
    predict(test_data)

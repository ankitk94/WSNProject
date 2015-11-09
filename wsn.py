import random
import math

# Generate Station
def generate_station(max_x, max_y, station_list):
    while True:
        coordinates = int(random.random() * max_x), int(random.random() * max_y)
        found_in_list = False
        for station in station_list:
            if station.get_station_coordinates == coordinates:
                found_in_list = True
                break
        if not found_in_list:
            return coordinates
        

def track_present():
    if random.random() > 0.3:
        return True
    return False
class RSU(object):

    rsu_id=0

    def __init__(self,station_1,station_2,distance_from_station1):
         self.station_id_1=station_1.get_station_id()
         self.station_id_2=station_2.get_station_id()
         self.id = RSU.rsu_id
         RSU.rsu_id += 1
         self.set_rsu_coordinate(distance_from_station1,station_1,station_2)
         
    def set_rsu_coordinate(self,distance_from_station1,station_1,station_2):
         coordinate_1 = station_1.get_station_coordinates()
         if station_1 == station_2:
             self.coordinates = coordinate_1
             return
         coordinate_2 = station_2.get_station_coordinates()
         r1=get_distance_between_stations(station_1.get_station_coordinates(),station_2.get_station_coordinates())-distance_from_station1
         r2=distance_from_station1
         self.coordinates =((1.0* (r1*coordinate_1[0]+ r2*coordinate_2[0]))/(r1+r2), (1.0*(r1*coordinate_1[1]+ r2*coordinate_2[1]))/(r1+r2))

    def deployed_on_station(self):
        if self.station_id_1 == self.station_id_2:
            return True
        return False
         
class Station(object):

    station_id = 0
    
    def __init__(self, max_x, max_y, station_list):
        self.station_coordinates = generate_station(max_x, max_y, station_list)
        self.station_id = Station.station_id
        Station.station_id += 1
        station_list.append(self)

    def get_station_id(self):
        return self.station_id

    def get_station_coordinates(self):
        return self.station_coordinates

    def set_rsu_id(self, rsu_id):
        self.rsu_id = rsu_id
        
class Train(object):

    train_id = 0

    def __init__(self, velocity, start_time, starting_station, destination_station, route, train_priority):
        self.id = Train.train_id
        Train.train_id += 1
        self.velocity = velocity
        self.start_time = start_time
        self.starting_station = starting_station
        self.destination_station = destination_station
        self.route = route
        self.between_stations = (starting_station, starting_station)
        self.distance_from_station1 = 0
        self.train_priority = train_priority
        self.train_running = False
        
class Event(object):

    event_id = 0
    
    def __init__(self, event_time, train_id, object_id, event_type):
        self.event_time = event_time
        self.train_id = train_id
        self.object_id = object_id
        self.event_type = event_type
        self.id = Event.event_id
        Event.event_id += 1


def get_distance_between_stations(coordinate1, coordinate2):
    return math.sqrt(1.0 * ((coordinate1[0] - coordinate2[0])**2) + 1.0 * ((coordinate1[1] - coordinate2[1])**2))

def initialize_adjacency_matrix(number_of_stations):
    adjacency_matrix = [[0.0 for x in range(number_of_stations)] for x in range(number_of_stations)]
    return adjacency_matrix

# Stations are the station id's
def generate_tracks_between_two_stations(station1, station2, adjacency_matrix_station):
    if track_present():
        adjacency_matrix_station[station1.get_station_id()][station2.get_station_id()] = adjacency_matrix_station[station2.get_station_id()][station1.get_station_id()] = get_distance_between_stations(station1.get_station_coordinates(), station2.get_station_coordinates())

def deploy_RSU_on_stations(station_list):
    deployed_RSU = []
    for station in station_list:
        deployed_RSU.append(RSU(station, station, 0))
        station.set_rsu_id(deployed_RSU[-1].id)
    return deployed_RSU

def deploy_RSU_between_two_stations(station1, station2, adjacency_matrix_station, distance_between_RSU):
    deployed_RSU = []
    distance_between_stations = adjacency_matrix_station[station1.get_station_id()][station2.get_station_id()]
    if distance_between_stations < distance_between_RSU:
        return deployed_RSU
    number_of_RSU_to_be_deployed = int(math.floor(distance_between_stations / distance_between_RSU))
    for RSU_count in range(number_of_RSU_to_be_deployed):
        deployed_RSU.append(RSU(station1, station2, (RSU_count+1) * distance_between_RSU))
    return deployed_RSU

def deploy_RSU(station_list, adjacency_matrix_station, distance_between_RSU):
    deployed_RSU = deploy_RSU_on_stations(station_list)
    for station1 in range(len(station_list)):
        for station2 in range(station1+1, len(station_list)):
            if adjacency_matrix_station[station1][station2] > 0:
                deployed_RSU = deployed_RSU + deploy_RSU_between_two_stations(station_list[station1],
                                                                          station_list[station2],
                                                                          adjacency_matrix_station,
                                                                          distance_between_RSU)
    return deployed_RSU

def generate_random_route(adjacency_matrix_station, current_station, count_station,
                          max_stations, visited_stations, generated_route):
    if count_station >= max_stations:
        return generated_route
    generated_route.append(current_station)
    visited_stations[current_station] = 1
    valid_next_stations = []
    for station in range(len(adjacency_matrix_station)):
        if visited_stations[station] == 0 and adjacency_matrix_station[current_station][station] > 0:
            valid_next_stations.append(station)
    if len(valid_next_stations) == 0:
        return generated_route
    # Randomly choose the station to visit
    station_to_visit = int(random.randint(0, len(valid_next_stations) - 1))
    #generated_route.append(valid_next_stations[station_to_visit])
    #visited_stations[valid_next_stations[station_to_visit]] = 1
    return generate_random_route(adjacency_matrix_station, valid_next_stations[station_to_visit],
                                 count_station + 1, max_stations, visited_stations, generated_route)
    
        
    
    

def generate_train_route(station_list, adjacency_matrix_station, max_route_length):
    # Generate length of the route randomly
    route_length = int((random.randint(2, max_route_length)))

    # Randomly assign a station as the origin station
    origin_station = int(math.floor((random.randint(0, len(station_list) - 1))))

    # Generate the route using DFS
    return generate_random_route(adjacency_matrix_station, origin_station, 0,
                                 route_length, [0 for i in range(len(adjacency_matrix_station))],
                                 [])            
                                 

def generate_train(station_list, adjacency_matrix_station, max_route_length, last_train_start_time):
    train_route = generate_train_route(station_list, adjacency_matrix_station,
                                       max_route_length)
    velocity = 1
    start_time = random.randint(last_train_start_time + 1, 3 + last_train_start_time)
    train_priority = 1
    train_generated = Train(velocity, start_time, train_route[0], train_route[-1],
                            train_route, train_priority)
    return (start_time, train_generated)
    

# Generate events of starting the traing
def generate_event_train_starting(station_list, adjacency_matrix_station,
                                  number_of_trains_to_be_generated, max_route_length):
    trains_generated = []
    start_time = 1
    for train_count in range(number_of_trains_to_be_generated):
        (start_time, generated_train) = generate_train(station_list, adjacency_matrix_station,
                                                       max_route_length, start_time)
        trains_generated.append(generated_train)
    return trains_generated


def custom_rsu_sorting(rsu_details1, rsu_details2):
    (rsu1, distance_from_station_1) = rsu_details1
    (rsu2, distance_from_station_2) = rsu_details2
    if distance_from_station_1 < distance_from_station_2:
        return -1
    return 1

# Generate events for trains meeting stations
def generate_events_train_meeting_stations(train, rsu_list, station_list, adjacency_matrix_station):
    route = train.route
    events = []
    station_crossing_time = train.start_time
    events.append(Event(station_crossing_time, train.id, route[0], 'Train starting'))
    rsu_in_route = []
    for station in range(len(route)-1):
        current_time = 0
        station_1_id = route[station]
        station_2_id = route[station+1]
        rsu_on_track = []
        for rsu in rsu_list:
            if (rsu.station_id_1 == station_1_id and rsu.station_id_2 == station_2_id) or (rsu.station_id_1 == station_2_id and rsu.station_id_2 == station_1_id):
                distance_from_station_1 = get_distance_between_stations(station_list[station_1_id].get_station_coordinates(),
                                                                            rsu.coordinates)
                    
                rsu_on_track.append((rsu, distance_from_station_1))
        # Sort the obtained list of RSU's on the track according to the distances
        # from station1
        rsu_on_track = sorted(rsu_on_track, cmp=custom_rsu_sorting)
        event_for_this_track = []
        for rsu in rsu_on_track:
            current_time = rsu[1] / (train.velocity * 1.0)
            generate_event = Event(current_time + station_crossing_time, train.id,
                                   rsu[0].id, 'RSU crossing')
            event_for_this_track.append(generate_event)
        events += event_for_this_track
        station_crossing_time += adjacency_matrix_station[route[station]][route[station+1]] /(1.0*train.velocity)
        generated_event = Event(station_crossing_time, train.id, route[station+1], 'Station crossing')
        events.append(generated_event)
    return events

def generate_events_all_trains_meeting_stations(train_list, rsu_list, station_list, adjacency_matrix_station):
    events = []
    for train in train_list:
        events += generate_events_train_meeting_stations(train, rsu_list, station_list, adjacency_matrix_station)
    return events


def sort_events(event1, event2):
    if event1.event_time < event2.event_time:
        return -1
    return 1

def generate_train_meeting_events(events, train_list, station_list, adjacency_matrix_station):
    all_events = []
    previous_event_time = 0
    for event in events:
        all_events.append(event)
        if event.event_type == 'Station crossing' or event.event_type == 'Train starting':
            current_event_time = event.event_time
            time_elapsed = current_event_time - previous_event_time
            previous_event_time=current_event_time
            # Skip updating for the train if the train is now starting
            for train in train_list:
                if current_event_time < train.start_time:
                    pass
                elif (train.id == event.train_id and event.event_type == 'Train starting'):
                    train.between_stations = (train.route[0], train.route[1])
                    train.train_running = True
                    for another_train in train_list:
                        # If this train is also in between the same stations
                        # They will meet only if their directions of movement is opposite
                        if train.between_stations == another_train.between_stations[::-1]:
                            # Calculate the meeting time
                            relative_velocity = train.velocity + another_train.velocity
                            meeting_time = ((adjacency_matrix_station[train.between_stations[0]][train.between_stations[1]] - another_train.distance_from_station1) * 1.0) / (1.0 * relative_velocity)
                            # Create and push the event
                            train_meeting_event = Event(current_event_time + meeting_time,
                                                        train.id, another_train.id,
                                                        'Train crossing')
                            all_events.append(train_meeting_event)
                else:
                    if train.train_running:
                        train.distance_from_station1 += 1.0 * train.velocity * time_elapsed
                if train.id == event.train_id and event.event_type == 'Station crossing':
                    # Update the stations
                    (station1, station2) = train.between_stations
                    train.distance_from_station1 = 0
                    current_route_index = train.route.index(station2)
                    if current_route_index +1 >= len(train.route):
                        train.train_running = False
                        train.between_stations = (station2, station2)
                    else:
                        train.between_stations = (train.route[current_route_index], train.route[current_route_index+1])
                        for another_train in train_list:
                            # If this train is also in between the same stations
                            # They will meet only if their directions of movement is opposite
                            if another_train.train_running and train.between_stations == another_train.between_stations[::-1]:
                                # Calculate the meeting time
                                relative_velocity = train.velocity + another_train.velocity
                                meeting_time = ((adjacency_matrix_station[train.between_stations[0]][train.between_stations[1]] - another_train.distance_from_station1) * 1.0) / (1.0 * relative_velocity)
                                # Create and push the event
                                train_meeting_event = Event(current_event_time + meeting_time,
                                                            train.id, another_train.id,
                                                            'Train crossing')
                                all_events.append(train_meeting_event)
    all_events = sorted(all_events, cmp=sort_events)
    return all_events

def testing(number_of_stations, max_x, max_y):
    list_of_stations = []
    for i in range(number_of_stations):
        Station(max_x, max_y, list_of_stations)
    adjacency_matrix_station = initialize_adjacency_matrix(number_of_stations)
    for i in range(number_of_stations):
        for j in range(i, number_of_stations):
            generate_tracks_between_two_stations(list_of_stations[i], list_of_stations[j], adjacency_matrix_station)
    distance_between_RSU = 8.0        
    rsu_list = deploy_RSU(list_of_stations, adjacency_matrix_station, distance_between_RSU)
    number_of_trains_to_be_generated = 20
    max_route_length = number_of_stations
    train_list = generate_event_train_starting(list_of_stations, adjacency_matrix_station,
                                  number_of_trains_to_be_generated, max_route_length)
    # Manage Events
    events = generate_events_all_trains_meeting_stations(train_list, rsu_list, list_of_stations, adjacency_matrix_station)
    events = sorted(events, cmp=sort_events)
    all_events = generate_train_meeting_events(events, train_list, list_of_stations, adjacency_matrix_station)
    return (list_of_stations, adjacency_matrix_station, rsu_list, all_events, train_list)

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
         self.currently_communicating_objects = 0
         RSU.rsu_id += 1
         self.crack_info_count = 0
         self.crack_list = []
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
        self.currently_communicating_objects = 0
        self.crack_info_count = 0
        self.crack_list = []
        
class Event(object):

    event_id = 0
    
    def __init__(self, event_time, train_id, object_id, event_type):
        self.event_time = event_time
        self.train_id = train_id
        self.object_id = object_id
        self.event_type = event_type
        self.id = Event.event_id
        Event.event_id += 1

class Crack(object):

    crack_id = 0

    def __init__(self, station1, station2):
        self.id = Crack.crack_id
        Crack.crack_id += 1
        self.station1 = station1
        self.station2 = station2

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


def get_train_object(train_id, train_list):
    for train in train_list:
        if train.id == train_id:
            return train

# Generate event list according to t-epsilon and t+epsilon
def generate_event_list_epsilon(event_list, epsilon, train_list):
    event_list_epsilon = []
    for event in event_list:
        start_time = event.event_time - epsilon
        train1_start_time = get_train_object(event.train_id, train_list).start_time
        train2_start_time = -1
        if event.event_type == "Train crossing":
            train2_start_time = get_train_object(event.object_id, train_list).start_time
        if start_time < train1_start_time or start_time < train2_start_time:
            start_time = max(train1_start_time, train2_start_time)
        end_time = event.event_time + epsilon
        event_list_epsilon.append((start_time, end_time, event))
    return event_list_epsilon

def custom_sort_epsilon_events(event1, event2):
    if event1[0] < event2[0]:
        return -1
    return 1

def custom_sort_epsilon_events_end_time(event1, event2):
    if event1[1] < event2[1]:
        return -1
    return 1

def get_rsu_object(rsu_id, rsu_list):
    for rsu in rsu_list:
        if rsu.id == rsu_id:
            return rsu

def get_tdma_slots_dfs(graph, mapping_to_graph, visited, vertex):
    if visited[vertex] == 1:
        return 0
    tdma_slots = 0
    connected_vertices = []
    visited[vertex] = 1
    for i in range(len(graph)):
        if graph[vertex][i] == 1:
            tdma_slots += 1
            connected_vertices.append(i)
        for v in connected_vertices:
            tdma_slots = max(tdma_slots, get_tdma_slots_dfs(graph, mapping_to_graph, visited, v))
    return tdma_slots
        
def find_connected_components(graph,visited,vertex):
    if visited[vertex]==1:
        return
    visited[vertex]=1
    for i in range(len(graph)):
        if graph[vertex][i] == 1:
            find_connected_components(graph,visited,i)
    return        


def get_object_from_graph(graph, index_of_graph, mapping_from_graph, train_list, rsu_list):
    #print mapping_from_graph
    #print index_of_graph
    #print "\n"
    for i in mapping_from_graph:
        if mapping_from_graph[i] == index_of_graph:
            if i[0] == 'rsu':
                for RSU in rsu_list:
                    if(RSU.id==i[1]):
                        
                        return ('rsu', RSU)
                raise Exception('Object not found in graph with given index in get_object_from_graph') 
            else:
                for train in train_list:
                    if(train.id==i[1]):
                        return ('train', train)
                raise Exception('Object not found in graph with given index in get_object_from_graph') 
    raise Exception('Object not found in graph with given index in get_object_from_graph')            

def packet_error():
    return False

def simulate_mac(graph,mapping_to_graph,connected_components,time_to_be_simulated, tdma_slots, train_list, rsu_list):
    #print "simulating mac"
    total_tdma_slots_for_simulation = int(time_to_be_simulated * 1.0 / tdma_slots)
    time_for_simulation_for_one_node = total_tdma_slots_for_simulation * 1.0 / (sum(connected_components))
    #print connected_components
    # Packet size in Bytes
    packet_size = 1024
    # Packet transfer speed Bytes/s
    transfer_speed = 20000 * 1024
    for node_index in range(len(connected_components)):
        if connected_components[node_index] == 1:
            #print "connected component present"
            node_object_ = get_object_from_graph(graph, node_index, mapping_to_graph, train_list, rsu_list)
            node_object = node_object_[1]
            # Total info size to send in Bytes
            total_information_to_send = 100 * 1024 * node_object.crack_info_count
            total_packets_to_send = int(total_information_to_send * 1.0 / packet_size)
            if total_packets_to_send == 0:
                continue
            
            total_packets_can_be_sent = int(transfer_speed * time_for_simulation_for_one_node / packet_size);
            packets_sent = 0
            #######
            #print "total packet can be sent " + str(total_packets_can_be_sent) 
            #print "total packet to send "+ str( total_packets_to_send)
            #####
            for packet in range(total_packets_can_be_sent):
                if not packet_error():
                    packets_sent += 1
            packets_sent=min(packets_sent,total_packets_to_send)
            total_cracks_info_received = int(packets_sent * node_object.crack_info_count * 1.0 / total_packets_to_send)
            #print "total crack info "
            #print total_cracks_info_received
            cracks_info_in_node = node_object.crack_list
            #####3
            #print "packet sent are"
            #print packets_sent
            for node_index_2 in range(len(connected_components)):
                if(node_index_2!=node_index and connected_components[node_index_2] == 1):
                    node_object_2_ = get_object_from_graph(graph, node_index_2, mapping_to_graph, train_list, rsu_list)
                    node_object_2=node_object_2_[1]
                    if node_object_2_[0] == 'train' and node_index_2 > 1:
                        #print mapping_to_graph
                        #print node_index_2
                        raise Exception('Train id > 1');
                    #print "inside inner for"
                    #print node_object_2_[0]
                   # print node_object_2_[1].id
                    if len(node_object.crack_list) >=total_cracks_info_received:
                        #print "sending the packet" 
                        
                        node_object_2.crack_list+=node_object.crack_list[-total_cracks_info_received:]
                        #print (node_object_2_[0])
                        #print node_index_2
                        #print type(node_object)
                        #print mapping_to_graph
                        node_object_2.crack_list=list(set(node_object_2.crack_list))
                        node_object_2.crack_info_count=len(node_object_2.crack_list)
                    else:
                        raise Exception('trying to send packets that do not exist in the crack array of node')
                        
                
            ##node_object_connected_to_node = get_object_from_graph(graph,  , mapping_from_graph, train_list, rsu_list)
            
    return 




    
def simulate_tdma(graph, mapping_to_graph, vertex, previous_start_time, new_start_time, train_list, rsu_list):
    visited = [0 for i in range(len(graph))]
    tdma_slots = 0
    tdma_slots = get_tdma_slots_dfs(graph, mapping_to_graph, visited, mapping_to_graph[vertex])
    ###finding the connected components
    visited = [0 for i in range(len(graph))]
    visited1=[0 for i in range(len(graph))]
    mark=[0 for i in range(len(graph))]
    while(True):
        all_visited=True
        for v in range(len(visited)):
            if(visited[v]==0):
                all_visited=False
                break
        if(all_visited):
            break
        
        find_connected_components(graph,visited,  v )
        for i in range(len(visited)):
            if(visited[i]+visited1[i]==1):
                mark[i]=1
                #print "in mark"
                #print (get_object_from_graph(graph, i, mapping_to_graph, train_list, rsu_list))[0]
            else:
                mark[i]=0       
        visited1 =[i for i in visited]
        #print "mark is \n"
        #print mark
        if(sum(mark)>1):
            
            simulate_mac(graph,mapping_to_graph,mark,new_start_time-previous_start_time, tdma_slots, train_list, rsu_list)
        else:
            pass

        
def simulate_epsilon_events(event_list_epsilon, train_list, rsu_list):
    current_events = []
    graph = [[0 for i in range(len(train_list) + len(rsu_list))] for p in range(len(train_list) + len(rsu_list))]
    mapping_to_graph = {}
    counter = 0
    detected_count=0
    undetected_count=0
    for train in train_list:
        mapping_to_graph[('train', train.id)] = counter
        counter += 1
    for rsu in rsu_list:
        mapping_to_graph[('rsu', rsu.id)] = counter
        counter += 1
    # Graph formation done
    previous_start_time = event_list_epsilon[0][0]
    for event in event_list_epsilon:
        if(event[2].event_type=='crack crossing'):
            print "event type is crack crossing"
            print event[2].id
            
            train_object=get_train_object(event[2].train_id,train_list)
            print train_object.id
            print train_object.crack_list
            print event[2].object_id
            if(event[2].object_id in train_object.crack_list):
                print "crack info found in train"
                detected_count+=1
            else:
                (train_object.crack_list).append(event[2].object_id)
                print "Crack list of " + str(train.id) + " " + str(train_object.crack_list)
                train_object.crack_info_count+=1
                undetected_count+=1 
        else:
            current_events.append(event)
            new_start_time = event[0]
            # Increase the count of communicating objects
            train = get_train_object(event[2].train_id, train_list)
            train.currently_communicating_objects += 1
            if event[2].event_type == "Train crossing":
                train = get_train_object(event[2].object_id, train_list)
                train.currently_communicating_objects += 1
                graph[mapping_to_graph[('train', event[2].train_id)]][mapping_to_graph[('train', train.id)]] = graph[mapping_to_graph[('train', train.id)]][mapping_to_graph[('train', event[2].train_id)]] = 1
            else:
                rsu = get_rsu_object(event[2].object_id, rsu_list)
                rsu.currently_communicating_objects += 1
                graph[mapping_to_graph[('train', event[2].train_id)]][mapping_to_graph[('rsu', rsu.id)]] = graph[mapping_to_graph[('rsu', rsu.id)]][mapping_to_graph[('train', event[2].train_id)]] = 1
            if previous_start_time == new_start_time and not(event[2].event_type == 'Train starting'):
                print event[2].event_type
                raise Exception('Same time')
            simulate_tdma(graph, mapping_to_graph, ('train', event[2].train_id), previous_start_time, new_start_time, train_list, rsu_list)
            current_events = sorted(current_events, cmp=custom_sort_epsilon_events_end_time)
            while event[0] > current_events[0][1]:
                current_event = current_events[0][2]
                current_events = current_events[1:]
                # Decrease the count of connected objects communicating with all of these objects
                train = get_train_object(current_event.train_id, train_list)
                train.currently_communicating_objects -= 1
                if train.currently_communicating_objects < 0:
                    raise Exception('Communicating objects should be >= 0')
                if current_event.event_type == "Train crossing":
                    train = get_train_object(current_event.object_id, train_list)
                    train.currently_communicating_objects -= 1
                    if graph[mapping_to_graph[('train', train.id)]][mapping_to_graph[('train', current_event.train_id)]] == 0:
                        raise Exception('Invalid graph')
                    graph[mapping_to_graph[('train', current_event.train_id)]][mapping_to_graph[('train', train.id)]] = graph[mapping_to_graph[('train', train.id)]][mapping_to_graph[('train', current_event.train_id)]] = 0
                    if train.currently_communicating_objects < 0:
                        raise Exception('Communicating objects should be >= 0')
                else:
                    rsu = get_rsu_object(current_event.object_id, rsu_list)
                    rsu.currently_communicating_objects -= 1
                    if graph[mapping_to_graph[('rsu', rsu.id)]][mapping_to_graph[('train', current_event.train_id)]] == 0:
                        raise Exception('Invalid graph')
                    graph[mapping_to_graph[('train', current_event.train_id)]][mapping_to_graph[('rsu', rsu.id)]] = graph[mapping_to_graph[('rsu', rsu.id)]][mapping_to_graph[('train', current_event.train_id)]] = 0
                    if rsu.currently_communicating_objects < 0:
                        raise Exception('Communicating objects should be >= 0')
            previous_start_time = new_start_time
    while len(current_events) > 0:
        new_start_time = current_events[0]
        current_event = current_events[0][2]
        current_events = current_events[1:]
        train = get_train_object(current_event.train_id, train_list)
        train.currently_communicating_objects -= 1
        if train.currently_communicating_objects < 0:
            raise Exception('Communicating objects should be >= 0')
        if current_event.event_type == "Train crossing":
            train = get_train_object(current_event.object_id, train_list)
            train.currently_communicating_objects -= 1
            if graph[mapping_to_graph[('train', train.id)]][mapping_to_graph[('train', current_event.train_id)]] == 0:
                raise Exception('Invalid graph')
            graph[mapping_to_graph[('train', current_event.train_id)]][mapping_to_graph[('train', train.id)]] = graph[mapping_to_graph[('train', train.id)]][mapping_to_graph[('train', current_event.train_id)]] = 0
            if train.currently_communicating_objects < 0:
                raise Exception('Communicating objects should be >= 0')
        else:
            rsu = get_rsu_object(current_event.object_id, rsu_list)
            rsu.currently_communicating_objects -= 1
            if rsu.currently_communicating_objects < 0:
                raise Exception('Communicating objects should be >= 0')
            if graph[mapping_to_graph[('rsu', rsu.id)]][mapping_to_graph[('train', current_event.train_id)]] == 0:
                raise Exception('Invalid graph')
            graph[mapping_to_graph[('train', current_event.train_id)]][mapping_to_graph[('rsu', rsu.id)]] = graph[mapping_to_graph[('rsu', rsu.id)]][mapping_to_graph[('train', current_event.train_id)]] = 0
            if rsu.currently_communicating_objects < 0:
                raise Exception('Communicating objects should be >= 0')
        previous_start_time = new_start_time
    print "detected count is "+ str(detected_count)
    print "undetected count is "+ str(undetected_count)



#######################################################current##################################################3
# to be filled : this is the TDMA part of the simulation that takes the list of simuntaneous events to be simulated



def simulate_mac_for_event_list(  simultaneous_events):
    time_slots=len(simultaneous_events)
    
    
    return 0
    

##############################################################################################33






#update the event list with the returned list
######################
def simulate_event(event_list,event_to_be_simulated):
    list_of_events_that_need_to_be_simulated_simuntaneously=[]
    updated_event_list=[]
    epsilon = 100
    if (event_to_be_simulated.event_type=='RSU crossing' or event_to_be_simulated.event_type=='Station crossing' or event_to_be_simulated.event_type=='Train starting'):
       
        for event in event_list:
            if event==event_to_be_simulated :
                if event in list_of_events_that_need_to_be_simulated_simuntaneously:
                    print "same if"
                list_of_events_that_need_to_be_simulated_simuntaneously.append(event)
            elif(event.event_time-event_to_be_simulated.event_time<=epsilon and event.event_time-event_to_be_simulated.event_time>0):
                    # either this rsu or this train is involved
                    if(event.event_type=='Train crossing' and (event.train_id==event_to_be_simulated.train_id or event.object_id ==event_to_be_simulated.train_id )):
                        if event in list_of_events_that_need_to_be_simulated_simuntaneously:
                            print "same if2"
                        list_of_events_that_need_to_be_simulated_simuntaneously.append(event)
                    if (event.event_type=='RSU crossing' or event.event_type=='Station crossing' or event.event_type=='Train starting'):
                        if(event.object_id==event_to_be_simulated.object_id):
                            if event in list_of_events_that_need_to_be_simulated_simuntaneously:
                                print "same if3"
                            list_of_events_that_need_to_be_simulated_simuntaneously.append(event)
                    ## when rsu crossing object id is the rsu    
                    
            else:
                updated_event_list.append(event)
        
    else:
        for event in event_list:
             if(event==simulated_event):
                        list_of_events_that_need_to_be_simulated_simuntaneously.append(event)
             elif(event.event_time-event_to_be_simulated.event_time<=epsilon and event.event_time-event_to_be_simulated.event_time>0):
                    if(event.event_type=='Train crossing'):
                        if(event.train_id==event_to_be_simulated.train_id or event.object_id==event_to_be_simulated.train_id ):
                            list_of_events_that_need_to_be_simulated_simuntaneously.append(event)

                        elif(event.train_id==event_to_be_simulated.object_id or event.object_id==event_to_be_simulated.object_id ):
                            list_of_events_that_need_to_be_simulated_simuntaneously.append(event)
                            
                   
                        
                    elif(event.event_type=='Station crossing' or event.event_type=='RSU crossing' or event_to_be_simulated.event_type=='Train starting'):
                        if(event.train_id==event_to_be_simulated.train_id or event.train_id==event_to_be_simulated.object_id ):
                            list_of_events_that_need_to_be_simulated_simuntaneously.append(event)
                                           
                    ## when rsu crossing object id is the rsu    
                    
             else:
                updated_event_list.append(event)
    statistics_of_mac = simulate_mac_for_event_list(  list_of_events_that_need_to_be_simulated_simuntaneously ,'Station or RSU crossing')    
    return (list_of_events_that_need_to_be_simulated_simuntaneously,updated_event_list)        

        
    






'''def custom_sort_epsilon_events(event1, event2):
    if event1[0] < event2[0]:
        return -1
    return 1
'''

def get_station_distance(station_1_id,station_2_id,station_list):
    stations=[]
    for station in station_list:
        if(station.station_id==station_1_id or station.station_id==station_2_id):
            stations.append(station)
        if(len(stations)==2):
            break
    distance = get_distance_between_stations(stations[0].station_coordinates,stations[1].station_coordinates)
    return distance





def contains_route(train_route, station_1, station_2):
    for i in range(len(train_route)-1):
        if (train_route[i] == station_1 and train_route[i+1] == station_2) or (train_route[i] == station_2 and train_route[i+1] == station_1):
            return True
    return False


def generate_crack_events(train_list, crack_list ,station_list):
    crack_events = []
    for crack in crack_list:
        crack=crack[1]
        # Find the route of crack
        # Find train for that route
        for train in train_list:
            print "Printing train"
            print train
            if contains_route(train.route, crack.station1, crack.station2):
                time=0
                for i in range(len(train.route)-1):
                    if(train.route[i]==crack.station1 and train.route[i+1]==crack.station2) or (train.route[i+1]==crack.station1 and train.route[i]==crack.station2):
                        ## time add
                        distance_between_station = get_station_distance(train.route[i],train.route[i+1],station_list)
                        if(crack.station1==train.route[i]):
                            time+=4
                            # change here for crack deployment
                        else:
                            time += distance_between_station -4

                        event = Event(  time, train.id, crack.id, 'crack crossing')
                        crack_events.append((time,time,event))
                        break
                             
                    else:
                        time+=get_station_distance(train.route[i],train.route[i+1])
                        
    return crack_events               
        


#######################333


def deploy_crack(train_list):
    crack_list=[]
    for train in train_list:
        route_of_train =train.route
        crack = Crack(route_of_train[0],route_of_train[1])
        crack_list.append( (train,crack) )
    return crack_list    


def testing(number_of_stations, max_x, max_y):
    list_of_stations = []
    for i in range(number_of_stations):
        Station(max_x, max_y, list_of_stations)
    adjacency_matrix_station = initialize_adjacency_matrix(number_of_stations)

    ## edited here
    adjacency_matrix_station[0][1]=adjacency_matrix_station[1][0]=40

    for i in range(number_of_stations):
        for j in range(i, number_of_stations):
            generate_tracks_between_two_stations(list_of_stations[i], list_of_stations[j], adjacency_matrix_station)
    distance_between_RSU = 8.0        
    rsu_list = deploy_RSU(list_of_stations, adjacency_matrix_station, distance_between_RSU)
    number_of_trains_to_be_generated = 2
    max_route_length = number_of_stations
    train_list = generate_event_train_starting(list_of_stations, adjacency_matrix_station,
                                  number_of_trains_to_be_generated, max_route_length)
    # Manage Events
    #edited here
    
    train_list[0].route =[0,1]
    train_list[1].route=[1,0]
    train_list[0].starting_station=train_list[0].route[0]
    train_list[1].starting_station=train_list[1].route[0]
    train_list[0].between_station=(train_list[0].route[0],train_list[0].route[0])
    train_list[1].between_station=(train_list[1].route[0],train_list[1].route[0])

    # deploying the crcks here
    crack_list = deploy_crack(train_list)
    ##########################################
    events = generate_events_all_trains_meeting_stations(train_list, rsu_list, list_of_stations, adjacency_matrix_station)
    events = sorted(events, cmp=sort_events)
    all_events = generate_train_meeting_events(events, train_list, list_of_stations, adjacency_matrix_station)
    """
    simultaneous_events_list = []

    
    (simultaneous,updated_event)=simulate_event(all_events,all_events[0])
    simultaneous_events_list.append(simultaneous)
    while len(updated_event) > 0:
        (simultaneous,updated_event)=simulate_event(updated_event,updated_event[0])
        simultaneous_events_list.append(simultaneous)
        
        """
    event_list_epsilon = generate_event_list_epsilon(all_events, 1, train_list)

    crack_events = generate_crack_events(train_list,crack_list,list_of_stations)
    event_list_epsilon += crack_events
    
    event_list_epsilon = sorted(event_list_epsilon, cmp=custom_sort_epsilon_events)
    simulate_epsilon_events(event_list_epsilon, train_list, rsu_list)
    
    """for train in train_list:
        print "for trains"
        if(len(train.crack_list)>0):
            print "train data here"
            print train.id
            #print "\n"
            print len(train.crack_list)"""
            
    """print "\n \n for RSU"
    for train in rsu_list:
        #print "llllllllllllllllllllllllllllllllll"
        if(len(train.crack_list)>0):
            print "rsu data here"
            print train.id
            #print "\n"
            print len(train.crack_list)
            
    #print "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"
    #print crack_list    """
    return (list_of_stations, adjacency_matrix_station, rsu_list, all_events, train_list,event_list_epsilon, crack_events, crack_list)
    for rsu in rsu_list:
        print "station1 station2 id"
        print str(rsu.station_id_1) + " " + str(rsu.station_id_2) + "  "+str(rsu.id)
    print train_list[0].route

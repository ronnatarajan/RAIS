import os 
import requests
import json
import streamlit as st
from backend.tools.logger import LoggerSetup
import time 

ROOT_REPO_DATA_SPECS = [
    {
        'url': 'https://api.github.com/repos/{org}/{repo}/contributors',
        'name': 'contributors',
        'type': list,
        'data_specs': {
            'members': {'action': 'count'}, 
            'member': {'action': 'extract', 'key': 'login'},  
        }
    },
    {
        'url': 'https://api.github.com/repos/{org}/{repo}/commits',
        'name': 'commits',
        'type': list,
        'data_specs': {
            'commits': {'action': 'count'}, 
            'commit_name': {'action': 'extract', 'key': ['commit', 'author', 'name'] },
            'commit_date': {'action': 'extract', 'key': ['commit', 'author', 'date'] },
        }
    }, 
    # {
    #     'url': 'https://api.github.com/repos/{org}/{repo}/releases',
    #     'name': 'releases',
    #     'type': list,
    #     'data_specs': {
    #         'releases': {'action': 'count'}, 
    #         'name': {'action': 'extract', 'key': ['name']},
    #         'tags': {'action': 'extract', 'key': ['tag_name']},
    #         'body': {'action': 'extract', 'key': ['body']},
    #     }
    # },  
    {
        'url': 'https://api.github.com/repos/{org}/{repo}',
        'name': 'repos',
        'type': dict,
        'data_specs': {
            'description': {'action': 'extract', 'key': 'description'},
            'stars': {'action': 'extract', 'key': 'stargazers_count'},
            'watchers': {'action': 'extract', 'key': 'subscribers_count'}, 
            'forks': {'action': 'extract', 'key': 'forks_count'},
            'issues': {'action': 'extract', 'key': 'open_issues_count'},
            'default_branch': {'action': 'extract', 'key': 'default_branch'},
            'owner_type': {'action': 'extract', 'key': ['owner', 'type']},
            'organization_type': {'action': 'extract', 'key': ['organization', 'type']},
            'owner': {'action': 'extract', 'key': ['owner', 'login']},
            'organization': {'action': 'extract', 'key': ['organization', 'login']},
            'has_wiki': {'action': 'extract', 'key': 'has_wiki'},
        }
    }, 
    {
        'url': 'https://api.github.com/repos/{org}/{repo}/pulls',
        'name': 'prs',
        'type': list,
        'data_specs': {
            'prs': {'action': 'count'}
        }
    }
]

MEMBER_REPO_DATA_SPECS = [ 
    {
        'url': 'https://api.github.com/repos/{org}/{repo}',
        'name': 'repos',  
        'type': dict,
        'data_specs': {
            'stars': {'action': 'extract', 'key': 'stargazers_count'},
            'forks': {'action': 'extract', 'key': 'forks_count'},
            'issues': {'action': 'extract', 'key': 'open_issues_count'},
            'watchers': {'action': 'extract', 'key': 'subscribers_count'},
        }
    }, 
    {
        'url': 'https://api.github.com/repos/{org}/{repo}/pulls',
        'name': 'prs',
        'type': list,
        'data_specs': {
            'prs': {'action': 'count'},
        }
    }
]

MEMBER_DATA_SPECS = [
    {
        'url': 'https://api.github.com/users/{user}/followers',
        'name': 'followers',
        'type': list, 
        'data_specs': {
            'followers': {'action': 'count'}, 
        }
    },
    {
        'url': 'https://api.github.com/users/{user}/repos',
        'name': 'repos',
        'type': list,  
        'data_specs': {
            'repos': {'action': 'extract', 'key': 'full_name'},  
            'repo_count': {'action': 'count'},  
        }
    }
]


class Metadata:
    def __init__(self, **repo_data):
        self.repo_data = repo_data
        logging_setup = LoggerSetup('Metadata')
        self.logger = logging_setup.get_logger()
        self.logger.info('Metadata object created')

    def agenerate(self): 
        self.logger.info('Generating root metadata...')
        self.root_metadata = self.get_root_metadata()
        yield self.root_metadata, 'metadata'
        self.logger.info('Root metadata generated.')

        self.logger.info('Generating organization metadata...')
        self.organization_metadata = self.get_organization_metadata()
        yield self.organization_metadata, 'organization'
        self.logger.info('Organization metadata generated.')

        self.logger.info('Generating owner metadata...')
        self.owner_metadata = self.get_owner_metadata()
        yield self.owner_metadata, 'owner'
        self.logger.info('Owner metadata generated.')

        self.logger.info('Generating member metadata...')
        self.member_metadata = self.get_member_metadata() 
        yield self.member_metadata, 'members'
        self.logger.info('Member metadata generated.')

    def get_root_metadata(self):
        self.logger.info('Processing root metadata...')
        return self.process_repo_data(ROOT_REPO_DATA_SPECS, org=self.repo_data['owner'], repo=self.repo_data['repo'])

    def get_organization_metadata(self):
        if self.root_metadata['organization']:
            self.logger.info('Processing organization metadata...')
            organization_metadata = {}
            organization_metadata['name'] = self.root_metadata['organization']
            organization_metadata.update(self.process_repo_data(MEMBER_DATA_SPECS, user=self.root_metadata['organization']))
            organization_metadata['metadata'] = self.process_user_repo_data(organization_metadata['repos'], 1)
            return organization_metadata
        else: 
            self.logger.info('Repo not affiliated with Organization...')
            return None

    def get_owner_metadata(self):
        self.logger.info('Processing owner metadata...')
        owner_metadata = {}
        if self.root_metadata['owner_type'] != 'Organization':
            owner_metadata['name'] = self.root_metadata['owner']
            owner_metadata.update(self.process_repo_data(MEMBER_DATA_SPECS, user=self.root_metadata['owner']))
            owner_metadata['metadata'] = self.process_user_repo_data(owner_metadata['repos'], 1)
        else: 
            owner_metadata = self.organization_metadata
        return owner_metadata

    def get_member_metadata(self):
        self.logger.info('Processing member metadata...')
        member_metadata = {}
        if int(self.root_metadata['members']) > 30: 
            self.logger.warning('Upperbound Threshold (30+ Members) Reached')
            member_metadata['error'] = 'Upperbound Threshold (30+ Members) Reached'
            return member_metadata

        for member in self.root_metadata['member'][:31]:  # Assuming 'member' should be 'members'
            try:
                member_metadata[member] = self.process_repo_data(MEMBER_DATA_SPECS, user=member)
                self.logger.info('Processed metadata for member: %s', member)
            except Exception as e:
                self.logger.error('Error processing member %s: %s', member, e)
                member_metadata[member] = None

        temp = {}
        for member in member_metadata: 
            try:
                temp[member] = self.process_user_repo_data(member_metadata[member]['repos'], 0)
                self.logger.info('Processed user repo data for member: %s', member)
            except Exception as e:
                self.logger.error('Error processing user repo data for member %s: %s', member, e)
                temp[member] = None

        for member in temp: 
            member_metadata[member]['metadata'] = temp[member]

        return member_metadata


    def extract_key(self, data, key):
        if isinstance(key, list):
            for part in key:
                if data is not None and part in data:
                    data = data[part]
                else:
                    return None
        else:
            data = data.get(key, None)
        return data

    def count(self, data):
        if isinstance(data, list):
            return len(data)
        elif isinstance(data, dict):
            return len(data.keys())
        else:
            return None

    def fetch_git_api(self, url):
        self.logger.info('Fetching %s', url)
        try: 
            try: 
                headers = {
                    "Accept": "application/vnd.github+json",
                    "Authorization": f"Bearer {st.session_state['GITHUB_AUTH_TOKEN']}",
                }
            except: 
                headers = {
                    "Accept": "application/vnd.github+json",
                    "Authorization": f"Bearer {os.environ.get('GITHUB_AUTH_TOKEN')}",
                }

            for _ in range(3):  # Retry up to 3 times
                try:
                    time.sleep(0.25)
                    response = requests.get(url, headers=headers, timeout=5)  # 5 seconds timeout
                    if response.status_code == 200:
                        return response.json()
                    else:
                        self.logger.error('Failed to fetch %s, status code: %s', url, response.status_code)
                        return None
                except requests.exceptions.Timeout:
                    self.logger.error('Timeout occurred while fetching %s. Retrying in 15 seconds...', url)
                    time.sleep(15)  # Wait for 15 seconds before retrying
            self.logger.error('Failed to fetch %s after 3 attempts', url)
            return None
        except Exception as e: 
            self.logger.error('An error occurred while fetching %s: %s', url, str(e))
            return None

    def process_repo_data(self, repo_data_specs, **formattable_vars):
        try: 
            headers = {
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {st.session_state['GITHUB_AUTH_TOKEN']}",
            }
        except: 
            headers = {
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {os.environ.get('GITHUB_AUTH_TOKEN')}",
            }

        results = {
            'error': {},
            'empty': []
        } 

        for spec in repo_data_specs:
            url = spec['url'].format(**formattable_vars)
            
            if spec['type'] == list: 
                i = 1
                url += "?state=all&page={i}&per_page=100"

                for key in spec['data_specs']:
                    if spec['data_specs'][key]['action'] == 'count':
                        results[key] = 0
                    elif spec['data_specs'][key]['action'] == 'extract':
                        results[key] = []

                # print(url.format(i=i))
                response = self.fetch_git_api(url.format(i=i))

                if not response:
                    if 'empty' in results:
                        results['empty'] += [spec['name']]
                    else: 
                        results['empty'] = [spec['name']]
                    response = [] 

                while len(response) and i <= 100:                    
                    for tag, detail in spec['data_specs'].items():
                        if detail['action'] == 'count':
                            results[tag] += self.count(response)
                        elif detail['action'] == 'extract':
                            if spec['type'] == list and isinstance(response, list):
                                results[tag] += [self.extract_key(item, detail['key']) for item in response]
                            else:
                                results[tag] += [self.extract_key(response, detail['key'])]

                        if i == 100 and tag in results: 
                            results['error'][tag] = f'Upperbound Threshold (10k+) Reached'

                    i += 1
                    # print(url.format(i=i))
                    response = self.fetch_git_api(url.format(i=i))
            else: 
                response = self.fetch_git_api(url)

                try: 
                    assert isinstance(response, spec['type']), f"Expected response type {spec['type'].__name__}, got {type(response).__name__} for url {url}."
                except: 
                    pass 

                if response is None or not isinstance(response, spec['type']):
                    continue

                for tag, detail in spec['data_specs'].items():
                    if detail['action'] == 'extract':
                        if spec['type'] == list and isinstance(response, list):
                            results[tag] = [self.extract_key(item, detail['key']) for item in response]
                        else:
                            results[tag] = self.extract_key(response, detail['key'])

        return results

    def process_user_repo_data(self, repos: list, type: int): 

        count_repo_metadata = {
            'error': {},
            'empty': {}
        } 

        if type == 1: 
            count = 100 
        else: 
            count = 50 

        for repo in repos[:count]:
            temp = self.process_repo_data(MEMBER_REPO_DATA_SPECS, org=repo.split('/')[0], repo=repo.split('/')[1])
            # print(count_repo_metadata)
            # print(temp)

            for key in temp:
                if key == 'error' and key in temp:
                        count_repo_metadata['error'][repo] = temp['error']
                elif key == 'empty' and key in temp: 
                        count_repo_metadata['empty'][repo] = temp['empty']
                else: 
                    if key not in count_repo_metadata:
                        if isinstance(temp[key], list): 
                            count_repo_metadata[key] = []
                        else:  
                            count_repo_metadata[key] = 0
                    # print(key)
                    count_repo_metadata[key] += temp[key]  
            # print(count_repo_metadata)

        if len(repos) > count: 
            count_repo_metadata['error'] = f'Upperbound Threshold ({str(count)}+ Repos) Reached'
            self.logger.warning(f'Upperbound Threshold ({str(count)}+ Repos) Reached')
            return count_repo_metadata
 

        return count_repo_metadata

-- Create participants table
create table if not exists public.participants (
    id uuid default uuid_generate_v4() primary key,
    participant_id integer,
    group_name text,
    censored_information text,
    feedback text,
    censorship_group text,
    condition text,
    experiment_start_time timestamp with time zone default timezone('utc'::text, now()) not null,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    updated_at timestamp with time zone default timezone('utc'::text, now()) not null,
    deleted_at timestamp with time zone
);

-- Add comments to the participants table
comment on table public.participants is 'Stores information about participants in experiments, including group details, feedback, and timestamps for various actions.';

-- Add comments to the columns in the participants table
comment on column public.participants.id is 'The unique identifier for each participant, generated using UUID.';
comment on column public.participants.participant_id is 'The participant number or identifier used in the experiment.';
comment on column public.participants.group_name is 'The name of the group the participant belongs to in the experiment.';
comment on column public.participants.censored_information is 'Any information that is censored for the participant.';
comment on column public.participants.feedback is 'Feedback provided by the participant during or after the experiment.';
comment on column public.participants.censorship_group is 'The censorship group assigned to the participant for the experiment.';
comment on column public.participants.condition is 'The experimental condition the participant is subjected to.';
comment on column public.participants.experiment_start_time is 'The timestamp when the experiment for the participant starts.';
comment on column public.participants.created_at is 'The timestamp when the participant record was created.';
comment on column public.participants.updated_at is 'The timestamp when the participant record was last updated.';
comment on column public.participants.deleted_at is 'The timestamp when the participant record was deleted, if applicable.';